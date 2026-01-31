import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
from ctgan.synthesizers.ctgan import CTGAN, Discriminator, Generator
from torch.utils.data import DataLoader, TensorDataset

class ClipCTGAN(CTGAN):
    """A CTGAN synthesizer with separate transformer training."""

    def __init__(self, max_grad_norm=1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_grad_norm = max_grad_norm
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
    def fit_transformer(self, data, discrete_columns):
        pass

    @random_state
    def sample(self, num_rows):
        """Sample data similar to the training data."""
        return super().sample(n=num_rows)

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

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

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])

        print(f'{self._verbose=}')
        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    optimizerD.zero_grad(set_to_none=False)
                    self._add_hooks(discriminator)

                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    #pen.backward(retain_graph=True)
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

                    total_norm = torch.cat(
                        [g.view(g.shape[0], -1) for g in per_sample_grads.values()], dim=1
                    ).norm(2, dim=1)
                    clip_factor = (self.max_grad_norm / (total_norm + 1e-6)).clamp(max=1.0)

                    for param, per_sample_grad in per_sample_grads.items():
                        clipped_grad = per_sample_grad * clip_factor.view(
                            -1, *([1] * (per_sample_grad.dim() - 1))
                        )
                        param.grad = clipped_grad.mean(dim=0)

                    optimizerD.step()
                    self._clear_hooks()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
