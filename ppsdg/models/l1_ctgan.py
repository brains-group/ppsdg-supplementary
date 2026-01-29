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

class L1CTGAN(CTGAN):
    """A CTGAN synthesizer with no training-by-sampling."""

    def __init__(self, **kwargs):
        """Create a DP-CTGAN synthesizer."""
        super().__init__(log_frequency=False, **kwargs)

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
