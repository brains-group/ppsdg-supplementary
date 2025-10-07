from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tabkit.utils import Configuration

DEFAULT_MLP_PARAMS = {
    "num_layers": 2,
    "d_token": 4,
    "n_head": 1,
    "factor": 32,
    "bias": True,
    "dim_t": 1024,
    "use_mlp": True,
    "d_numerical": 6,
    "categories": [3, 10, 17, 8, 16, 7, 6, 3, 43],
}

DEFAULT_DIFFUSION_PARAMS = {
    "num_timesteps": 50,
    "scheduler": "power_mean_per_column",
    "cat_scheduler": "log_linear_per_column",
    "noise_dist": "uniform_t",
    "sampler_params": {"stochastic_sampler": True, "second_order_correction": True},
    "edm_params": {"precond": True, "sigma_data": 1.0, "net_conditioning": "sigma"},
    "noise_dist_params": {"P_mean": -1.2, "P_std": 1.2},
    "noise_schedule_params": {
        "sigma_min": 0.002,
        "sigma_max": 80,
        "rho": 7,
        "eps_max": 0.001,
        "eps_min": 1e-05,
        "rho_init": 7.0,
        "rho_offset": 5.0,
        "k_init": -6.0,
        "k_offset": 1.0,
    },
}

DEFAULT_TRAIN_PARAMS = {
    "steps": 8000,
    "lr": 0.001,
    "weight_decay": 0,
    "ema_decay": 0.997,
    "batch_size": 4096,
    "check_val_every": 2000,
    "lr_scheduler": "reduce_lr_on_plateau",
    "factor": 0.9,
    "reduce_lr_patience": 50,
    "closs_weight_schedule": "anneal",
    "c_lambda": 1.0,
    "d_lambda": 1.0,
}


DEFAULT_DATA_PARAMS = {
    "dequant_dist": "none",
    "int_dequant_factor": 0.0,
}


# this is basically mimicing the argparse in the original script.
@dataclass
class TabDiffArgs:
    no_wandb: bool = True
    mode: str = "train"
    debug: bool = False
    deterministic: bool = False
    y_only: bool = False
    non_learnable_schedule: bool = False
    num_samples_to_generate: int | None = None
    ckpt_path: str | None = None
    report: bool = False
    num_runs: int = 20
    impute: bool = False
    trial_start: int = 0
    trial_size: int = 50
    resample_rounds: int = 1
    impute_condition: str = "x_t"
    y_only_model_path: str | None = None
    w_num: float = 0.6
    w_cat: float = 0.6


class TabDiffConfig(Configuration):
    exp_name: str
    dataset_name: str
    dataset_info: dict
    data_dir: str
    model_save_path_base: str
    result_save_path_base: str
    n_cont: int
    cat_cardinalities: np.ndarray
    mlp_params: dict | None = None
    diffusion_params: dict | None = None
    train_params: dict | None = None
    sample_batch_size: int = 4096
    device: str = "cpu"
    args: TabDiffArgs | None = None
    data_params: dict | None = None

    """
    Monkey patch to get tabdiff to work with everything else.
    params related to the training procedure are grouped under train_params.
    params related to the model architecture are grouped under (mlp|diffusion)_params.

    What I could find from their code:

    train_params:
        - lr: float
        - weight_decay: float
        - steps: int
        - batch_size: int
        - check_val_every: int
    diffusion_params:
        - num_timesteps: int = 1000
        - scheduler: str = "power_mean"
        - cat_scheduler: str = "log_linear"
        - noise_dist: str = "uniform"
        - noise_schedule_params: dict
            - rho
            - k
            - eps_max
            - eps_min
            - k_init
            - k_offset
            - sigma_max
            - sigma_min
        - edm_params
            - sigma_data: float
            - precond: bool
            - net_conditioning: str
        - noise_dist_params: dict
            - P_std
            - P_mean
        - sampler_params: dict
            stochastic_sampler: bool
            second_order_correction: bool
    """

    def __post_init__(self):
        self.train_params = {**DEFAULT_TRAIN_PARAMS, **(self.train_params or {})}
        self.mlp_params = {**DEFAULT_MLP_PARAMS, **(self.mlp_params or {})}
        self.diffusion_params = {**DEFAULT_DIFFUSION_PARAMS, **(self.diffusion_params or {})}
        self.data_params = {**DEFAULT_DATA_PARAMS, **(self.data_params or {})}
        if self.args is None:
            self.args = TabDiffArgs()
        # we need to compute this once and keep it, because the main script modifies the config!! (the args when we do inference)
        # that will change the hash and thus the result of get_unique_name()
        self.model_save_path = str(
            (Path(self.model_save_path_base) / self.get_unique_name()).resolve()
        )
        self.result_save_path = str(
            (Path(self.result_save_path_base) / self.get_unique_name()).resolve()
        )

    # @property
    # def model_save_path(self) -> str:
    #     return str((Path(self.model_save_path_base) / self.get_unique_name()).resolve())
    #
    # @property
    # def result_save_path(self) -> str:
    #     return str(
    #         (Path(self.result_save_path_base) / self.get_unique_name()).resolve()
    #     )
