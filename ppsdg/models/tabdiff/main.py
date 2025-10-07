"""
Basically a paraphrase of the tabdiff/main.py file.
"""

import os
import pickle
from pathlib import Path
from typing import Literal

import torch
import wandb
from torch.utils.data import DataLoader

from .config_tabdiff import TabDiffConfig
from .metrics import TabMetrics
from .models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from .modules.main_modules import Model, UniModMLP
from .trainer import Trainer
from .utils_train import TabDiffDataset

TABDIFF_CACHE_DIR = Path(".tabdiff/")


def build_trainer(config: TabDiffConfig):
    # replace with the argparse here.
    args = config.args

    # TODO: make the tabdiff dataset and loaders here
    train_data = TabDiffDataset(
        config.dataset_name,
        config.data_dir,
        config.dataset_info,
        y_only=args.y_only,
        isTrain=True,
        dequant_dist=config.data_params["dequant_dist"],
        int_dequant_factor=config.data_params["int_dequant_factor"],
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.train_params["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    d_numerical, categories = train_data.d_numerical, train_data.categories

    val_data = TabDiffDataset(
        config.dataset_name,
        config.data_dir,
        config.dataset_info,
        y_only=args.y_only,
        isTrain=False,
        dequant_dist=config.data_params["dequant_dist"],
        int_dequant_factor=config.data_params["int_dequant_factor"],
    )

    real_data_path = TABDIFF_CACHE_DIR / f"synthetic/{config.dataset_name}/real.csv"
    test_data_path = TABDIFF_CACHE_DIR / f"synthetic/{config.dataset_name}/test.csv"
    val_data_path = TABDIFF_CACHE_DIR / f"synthetic/{config.dataset_name}/val.csv"
    if not os.path.exists(val_data_path):
        print(
            f"{config.dataset_name} does not have its validation set. During MLE evaluation, a validation set will be splitted from the training set!"
        )
        val_data_path = None
    metric_list = ["density"]

    metrics = TabMetrics(
        real_data_path,
        test_data_path,
        val_data_path,
        config.dataset_info,
        config.device,
        metric_list=metric_list,
    )

    ## Load the module and models
    config.mlp_params["d_numerical"] = config.n_cont
    config.mlp_params["categories"] = (
        config.cat_cardinalities + 1
    ).tolist()  # add one for the mask category
    if args.y_only:
        config.mlp_params["use_mlp"] = (
            False  # drop the mlp when training the unconditional model
        )
        config.mlp_params["dim_t"] = 128  # reduce the size of the mlp
        main_model_path = args.ckpt_path
        if main_model_path is None:
            main_model_parent_path = (
                TABDIFF_CACHE_DIR
                / f"ckpt/{config.dataset_name}/{config.exp_name.replace('_y_only', '')}"
            )
            main_model_path_arr = list(main_model_parent_path.glob("best_ema_model*"))
            assert main_model_path_arr, (
                f"Cannot not infer the main model's ckpt_path from {main_model_parent_path}, please make sure that you first train a main model before training the y_only model!"
            )
            main_model_path = main_model_path_arr[0]
        main_model_configs = pickle.load(
            open(os.path.join(os.path.dirname(main_model_path), "config.pkl"), "rb")
        )
        if (
            main_model_configs["config.diffusion_params"]["scheduler"]
            == "power_mean_per_column"
        ):  # if learnable schedule is enabled in the main model, we need to infer noise params of the target column from the main model ckpt and train the y_only model with those params
            from tabdiff.models.noise_schedule import (
                LogLinearNoise_PerColumn,
                PowerMeanNoise_PerColumn,
            )

            if config.dataset_info["task_type"] == "regression":
                noise_schedule = PowerMeanNoise_PerColumn(
                    num_numerical=main_model_configs["unimodconfig.mlp_params"][
                        "d_numerical"
                    ],
                    **main_model_configs["config.diffusion_params"][
                        "noise_schedule_params"
                    ],
                )
                config.diffusion_params["noise_schedule_params"]["rho"] = (
                    noise_schedule.rho()[0].item()
                )  # the target col is placed at the first position
            else:
                noise_schedule = LogLinearNoise_PerColumn(
                    num_categories=len(
                        main_model_configs["unimodconfig.mlp_params"]["categories"]
                    ),
                    **main_model_configs["config.diffusion_params"][
                        "noise_schedule_params"
                    ],
                )
                config.diffusion_params["noise_schedule_params"]["k"] = (
                    noise_schedule.k()[0].item()
                )  # the target col is placed at the first position

    backbone = UniModMLP(**config.mlp_params)
    model = Model(backbone, **config.diffusion_params["edm_params"])
    model.to(config.device)

    ## Create and load y_only_model for imputation
    y_only_model = None
    if args.impute:
        y_only_model_path = args.y_only_model_path
        if y_only_model_path is None:
            y_only_model_parent_path = (
                TABDIFF_CACHE_DIR
                / f"ckpt/{config.dataset_name}/{config.exp_name}_y_only"
            )
            y_only_model_path_arr = list(
                y_only_model_parent_path.glob("best_ema_model*")
            )
            assert y_only_model_path_arr, (
                f"Cannot not infer y_only model's ckpt_path from {y_only_model_parent_path}, please make sure that you first train a y_only model before testing imputation!"
            )
            y_only_model_path = y_only_model_path_arr[0]
        y_only_model_config_path = os.path.join(
            os.path.dirname(y_only_model_path), "config.pkl"
        )
        with open(y_only_model_config_path, "rb") as f:
            y_only_model_config = pickle.load(f)
        y_only_model = UniModMLP(**y_only_model_config["unimodconfig.mlp_params"])
        y_only_model = Model(
            y_only_model, **y_only_model_config["config.diffusion_params"]["edm_params"]
        )
        y_only_model.to(config.device)
        # load weights
        state_dicts = torch.load(y_only_model_path, map_location=config.device)
        y_only_model.load_state_dict(state_dicts["denoise_fn"])

    if not args.y_only and not args.non_learnable_schedule:
        config.diffusion_params["scheduler"] = "power_mean_per_column"
        config.diffusion_params["cat_scheduler"] = "log_linear_per_column"
    diffusion = UnifiedCtimeDiffusion(
        num_classes=config.cat_cardinalities,
        num_numerical_features=config.n_cont,
        denoise_fn=model,
        y_only_model=y_only_model,
        **config.diffusion_params,
        device=config.device,
    )
    num_params = sum(p.numel() for p in diffusion.parameters())
    print("The number of parameters = ", num_params)
    diffusion.to(config.device)
    diffusion.train()

    ## Print the configs
    # printed_configs = json.dumps(
    #     raw_config, default=lambda x: int(x) if isinstance(x, np.int64) else x, indent=4
    # )
    # print(f"The config of the current run is : \n {printed_configs}")

    ## Enable Wandb
    project_name = f"tabdiff_{config.dataset_name}"
    logger = wandb.init(
        project="tabdiff",
        name=config.exp_name,
        # config=raw_config,
        mode="disabled" if args.debug or args.no_wandb else "online",
    )

    ## Load Trainer
    # sample_batch_size = raw_config["sample"]["batch_size"]
    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        metrics,
        logger,
        **config.train_params,
        sample_batch_size=config.sample_batch_size,
        num_samples_to_generate=args.num_samples_to_generate,
        model_save_path=config.model_save_path,
        result_save_path=config.result_save_path,
        device=config.device,
        ckpt_path=config.args.ckpt_path,
        y_only=args.y_only,
    )
    Path(config.model_save_path).mkdir(exist_ok=True, parents=True)
    Path(config.result_save_path).mkdir(exist_ok=True, parents=True)
    print("we made ", config.model_save_path, " and ", config.result_save_path)
    return trainer
