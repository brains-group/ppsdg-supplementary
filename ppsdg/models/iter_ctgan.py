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
from torch.func import functional_call, vmap, grad

class IterCTGAN(CTGAN):
    def __init__(
        self,
        log_frequency=True,
        max_grad_norm=1.0,
        use_gradient_penalty=True,
        gp_lambda=10,
        **kwargs
    ):
        """Create a DP-CTGAN synthesizer."""
        super().__init__(
            pac=1,
            log_frequency=log_frequency,
            **kwargs
        )
        self.max_grad_norm = max_grad_norm
        self.gp_lambda = gp_lambda

    @random_state
    def fit_transformer(self, data, discrete_columns):
        self._validate_discrete_columns(data, discrete_columns)
        self._validate_null_data(data, discrete_columns)

        self._transformer = DataTransformer()
        self._transformer.fit(data, discrete_columns)

    @random_state
    def sample(self, num_rows):
        """Sample data similar to the training data."""
        return super().sample(n=num_rows)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        """
        random_seeds = torch.empty(2, dtype=int).random_()
        loader_gen = torch.Generator()
        loader_gen.manual_seed(random_seeds[0].item())
        noise_gen = torch.Generator(device=self._device)
        noise_gen.manual_seed(random_seeds[1].item())

        train_data = self._transformer.transform(train_data)
        data_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_data.astype("float32"))),
            batch_size=self._batch_size, shuffle=True, drop_last=True,
            generator=loader_gen
        )

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

        epoch_iterator = tqdm(range(self._epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            #for batch_data in data_loader:
            #    batch_size = batch_data[0].shape[0]
            for id_ in range(steps_per_epoch):
                batch_size = self._batch_size

                # Discriminator
                c1, m1, col, opt = self._data_sampler.sample_condvec(batch_size)
                c1 = torch.from_numpy(c1).to(self._device)

                fakez = torch.normal(mean=mean, std=std, generator=noise_gen)
                fakez = torch.cat([fakez, c1], dim=1)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fake_cat = torch.cat([fakeact, c1], dim=1)

                #real_data = batch_data[0].to(self._device)
                real_data = self._data_sampler.sample_data(train_data, batch_size, col, opt)
                real_data = torch.from_numpy(real_data.astype('float32')).to(self._device)
                real_cat = torch.cat([real_data, c1], dim=1)

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                alpha = torch.rand(real_cat.size(0), 1, device=self._device)
                alpha = alpha.repeat(1, real_cat.size(1))
                interpolates = alpha * real_cat + ((1 - alpha) * fake_cat)
                disc_interpolates = discriminator(interpolates)
                interpolate_gradients, = torch.autograd.grad(
                    outputs=disc_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones(disc_interpolates.size(), device=self._device),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )
                grad_penalties = (interpolate_gradients.norm(2, dim=1) - 1) ** 2 * self.gp_lambda

                accum_grads = [torch.zeros_like(p) for p in discriminator.parameters()]

                for j in range(batch_size):
                    optimizerD.zero_grad(set_to_none=False)
                    loss_tmp = y_fake[j] - y_real[j] + grad_penalties[j]
                    loss_tmp.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.max_grad_norm)
                    for k, p in enumerate(discriminator.parameters()):
                        accum_grads[k] += p.grad

                for j, p in enumerate(discriminator.parameters()):
                    p.grad = accum_grads[j] / (2*batch_size)

                optimizerD.step()

                loss_d = (y_fake.mean() - y_real.mean()) / 2 + grad_penalties.mean()
                total_norm = torch.cat([p.grad.flatten() for p in discriminator.parameters()]).norm(2).item()
                #print(i, loss_d.item(), total_norm)

                # Generator
                fakez = torch.normal(mean=mean, std=std, generator=noise_gen)
                c1, m1, col, opt = self._data_sampler.sample_condvec(self._batch_size)

                c1 = torch.from_numpy(c1).to(self._device)
                m1 = torch.from_numpy(m1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fake_cat = torch.cat([fakeact, c1], dim=1)
                y_fake = discriminator(fake_cat)

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
                'Discriminator Grad Norm': [total_norm],
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
