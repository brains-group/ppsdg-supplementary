"""DP-TVAE Synthesizer module.

This module provides a differentially private version of the TVAE synthesizer
using a manual, hook-based implementation of DP-SGD for efficient per-sample
gradient computation.
"""

import numpy as np
import pandas as pd
import torch
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.tvae import TVAE, Decoder, Encoder, _loss_function
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ppsdg.models.dp_transformer import DPDataTransformer


class DPTVAE(TVAE):
    """A TVAE synthesizer with a manual, efficient DP-SGD implementation."""

    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        **kwargs,
    ):
        """Create a DP-TVAE synthesizer.

        Args:
            epsilon (float):
                Target epsilon for the (epsilon, delta)-DP guarantee.
            delta (float):
                Target delta for the (epsilon, delta)-DP guarantee.
            max_grad_norm (float):
                Clipping threshold for per-sample gradients.
            **kwargs:
                Other keyword arguments for the original ``TVAE`` synthesizer.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        # Attributes for hook-based per-sample gradient computation
        self._activations = {}
        self._grad_outputs = {}
        self._hooks = []

    def _add_hooks(self, model, prefix):
        """Adds forward and backward hooks to the model's linear layers."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                key = f"{prefix}.{name}"
                self._hooks.append(
                    module.register_forward_hook(self._capture_activation(key))
                )
                self._hooks.append(
                    module.register_full_backward_hook(self._capture_grad_output(key))
                )

    def _capture_activation(self, key):
        def hook(module, input, output):
            self._activations[key] = input[0].detach()
        return hook

    def _capture_grad_output(self, key):
        def hook(module, grad_input, grad_output):
            self._grad_outputs[key] = grad_output[0].detach()
        return hook

    def _clear_hooks(self):
        """Removes all registered hooks and clears data."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()
        self._grad_outputs.clear()

    @random_state
    def fit_transformer(self, full_data, discrete_columns):
        self.transformer = DPDataTransformer()
        self.transformer.fit(full_data, discrete_columns)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the DPTVAE model to the training data using efficient DP-SGD."""
        if self.transformer is None:
            warnings.warn("Transformer was not preconstructed--likely privacy leak.", UserWarning)
            self.fit_transformer(train_data, discrete_columns)

        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(
            torch.from_numpy(train_data.astype("float32")).to(self._device)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)

        # NOTE: inwon -- is Adam OK? Need to think about this...
        optimizer = Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            weight_decay=self.l2scale
        )

        noise_multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon if self.epsilon > 0 else 0.0

        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        loss_values = []

        for i in iterator:
            for id_, data in enumerate(loader):
                optimizer.zero_grad()
                self._clear_hooks()

                batch_size = data[0].shape[0]
                real = data[0].to(self._device)

                self._add_hooks(encoder, 'encoder')
                self._add_hooks(decoder, 'decoder')

                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = decoder(emb)

                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2

                loss.backward()

                per_sample_grads = {}
                # --- Encoder Grads ---
                for name, module in encoder.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        key = f"encoder.{name}"
                        activation = self._activations[key]
                        grad_output = self._grad_outputs[key]
                        grad_w = torch.einsum('ni,nj->nij', grad_output, activation)
                        per_sample_grads[module.weight] = grad_w
                        if module.bias is not None:
                            grad_b = grad_output
                            per_sample_grads[module.bias] = grad_b

                # --- Decoder Grads ---
                for name, module in decoder.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        key = f"decoder.{name}"
                        activation = self._activations[key]
                        grad_output = self._grad_outputs[key]
                        grad_w = torch.einsum('ni,nj->nij', grad_output, activation)
                        per_sample_grads[module.weight] = grad_w
                        if module.bias is not None:
                            grad_b = grad_output
                            per_sample_grads[module.bias] = grad_b

                total_norm = torch.cat([g.view(batch_size, -1) for g in per_sample_grads.values()], dim=1).norm(2, dim=1)
                clip_factor = (self.max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)

                for param, per_sample_grad in per_sample_grads.items():
                    clipped_grad = per_sample_grad * clip_factor.view(-1, *([1] * (per_sample_grad.dim() - 1)))
                    total_grad = clipped_grad.sum(dim=0)
                    noise = torch.normal(
                        0,
                        noise_multiplier * self.max_grad_norm,
                        size=total_grad.shape,
                        device=self._device,
                    )
                    param.grad = (total_grad + noise) / batch_size

                optimizer.step()
                decoder.sigma.data.clamp_(0.01, 1.0)

                loss_values.append([i, id_, loss.detach().cpu().item()])

        self.loss_values = pd.DataFrame(loss_values, columns=['Epoch', 'Batch', 'Loss'])

        self.encoder = encoder
        self.decoder = decoder
        self.encoder.eval()
        self.decoder.eval()
        self._clear_hooks()

    @random_state
    def sample(self, num_rows):
        return super().sample(num_rows)
