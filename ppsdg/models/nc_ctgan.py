"""DP-CTGAN Synthesizer module.

This module provides a differentially private version of the CTGAN
synthesizer. It uses a manual implementation of DP-SGD (Differentially
Private Stochastic Gradient Descent) to train the discriminator. This
implementation uses hooks for efficient per-sample gradient computation.
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ctgan.data_sampler import DataSampler
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.ctgan import CTGAN, Discriminator, Generator
from torch.utils.data import DataLoader, TensorDataset

from ppsdg.models.dp_transformer import DPDataTransformer


class DPDiscriminator(Discriminator):
    """
    Discriminator that adds a sigmoid activation for BCE loss. We need to
    use BCE loss for DP, because we can't bound the sensitivity of Wasserstein
    loss. Wasserstein loss is problematic because it can be unboundedly large
    if the discriminator is too good. This means that the amount of noise we
    need to add to the gradients to ensure differential privacy can also be
    unboundedly large. This makes it impossible to provide any meaningful
    privacy guarantees.
    """

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return torch.sigmoid(self.seq(input_.view(-1, self.pacdim)))


class NCNoDPCTGAN(CTGAN):
    """A CTGAN synthesizer with a manual, efficient DP-SGD implementation."""

    def __init__(
        self,
        epsilon=None,
        delta=None,
        max_grad_norm=None,
        use_bce_loss=True,
        **kwargs,
    ):
        """Create a DP-CTGAN synthesizer."""
        super().__init__(pac=1, **kwargs)
        self.use_bce_loss = use_bce_loss

        self._activations = {}
        self._grad_outputs = {}
        self._hooks = []

    def _add_hooks(self, discriminator):
        """Adds forward and backward hooks to the discriminator's linear layers."""
        for name, module in discriminator.named_modules():
            if isinstance(module, torch.nn.Linear):
                self._hooks.append(
                    module.register_forward_hook(self._capture_activation(name))
                )
                self._hooks.append(
                    module.register_full_backward_hook(self._capture_grad_output(name))
                )

    def _capture_activation(self, name):
        def hook(module, input, output):
            self._activations[name] = input[0].detach()

        return hook

    def _capture_grad_output(self, name):
        def hook(module, grad_input, grad_output):
            self._grad_outputs[name] = grad_output[0].detach()

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
        self._validate_discrete_columns(full_data, discrete_columns)
        self._validate_null_data(full_data, discrete_columns)

        self._transformer = DPDataTransformer()
        self._transformer.fit(full_data, discrete_columns)

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the synthesizer models to the training data using DP-SGD."""
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        if self.use_bce_loss:
            warnings.warn("Using BCE loss.", UserWarning)
        else:
            warnings.warn("Using Wasserstein loss.", UserWarning)

        if epochs is None:
            epochs = self._epochs

        if self._transformer is None:
            warnings.warn("Transformer was not preconstructed--likely privacy leak.", UserWarning)
            self.fit_transformer(train_data, discrete_columns)

        transformed_data = self._transformer.transform(train_data)

        dataset = TensorDataset(torch.from_numpy(transformed_data.astype("float32")))
        data_loader = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True, drop_last=True
        )

        self._data_sampler = DataSampler(
            transformed_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = DPDiscriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

        # NOTE: same deal here.
        optimizerG = torch.optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        optimizerD = torch.optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))
        loss_values = []

        for i in epoch_iterator:
            for batch_data in data_loader:
                optimizerD.zero_grad(set_to_none=False)
                self._clear_hooks()
                self._add_hooks(discriminator)

                batch_size = batch_data[0].shape[0]
                real_data = batch_data[0].to(self._device)

                fakez = torch.normal(
                    mean=torch.zeros(batch_size, self._embedding_dim), std=1
                ).to(self._device)
                condvec = self._data_sampler.sample_condvec(batch_size)
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fake_cat = torch.cat([fakeact, c1], dim=1)

                c2 = c1[np.random.permutation(batch_size)]
                real_cat = torch.cat([real_data, c2], dim=1)

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                if self.use_bce_loss:
                    loss_real = F.binary_cross_entropy(y_real, torch.ones_like(y_real))
                    loss_fake = F.binary_cross_entropy(y_fake, torch.zeros_like(y_fake))
                    loss_d = (loss_real + loss_fake) / 2
                else:
                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = torch.mean(y_fake) - torch.mean(y_real) + pen

                loss_d.backward()

                per_sample_grads = {}
                for name, module in discriminator.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        activation = self._activations[name]
                        grad_output = self._grad_outputs[name]
                        grad_w = torch.einsum("ni,nj->nij", grad_output, activation)
                        per_sample_grads[module.weight] = grad_w
                        if module.bias is not None:
                            grad_b = grad_output
                            per_sample_grads[module.bias] = grad_b

                for param, per_sample_grad in per_sample_grads.items():
                    param.grad = per_sample_grad.mean(dim=0)
                    torch.normal(
                        0,
                        0,
                        size=per_sample_grad.shape[1:],
                        device=self._device,
                    )

                optimizerD.step()

                # --- Train Generator ---
                optimizerG.zero_grad(set_to_none=False)
                fakez = torch.normal(
                    mean=torch.zeros(self._batch_size, self._embedding_dim), std=1
                ).to(self._device)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                m1 = torch.from_numpy(m1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                loss_g.backward()
                optimizerG.step()


                generator_loss = loss_g.detach().cpu().item()
                discriminator_loss = loss_d.detach().cpu().item()

                loss_values.append([i, generator_loss, discriminator_loss])

                if self._verbose:
                    epoch_iterator.set_description(
                        description.format(gen=generator_loss, dis=discriminator_loss)
                    )

        self.loss_values = pd.DataFrame(loss_values, columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])

        self._clear_hooks()

    @random_state
    def sample(self, num_rows):
        """Sample data similar to the training data."""
        return super().sample(n=num_rows)
