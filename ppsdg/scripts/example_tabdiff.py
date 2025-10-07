# %%


from tabkit import DatasetConfig, TableProcessor, TableProcessorConfig

from ppsdg.models.tabdiff import train_tabdiff, eval_tabdiff

dataset_config = DatasetConfig.from_yaml("config/dataset/adult.yaml")
proc = TableProcessor(
    dataset_config=dataset_config,
    config=TableProcessorConfig(),
).prepare()


train_params = {
    "steps": 1,
    "lr": 0.001,
    "weight_decay": 0,
    "ema_decay": 0.997,
    "batch_size": 4096,
    "check_val_every": 1,
    "lr_scheduler": "reduce_lr_on_plateau",
    "factor": 0.9,
    "reduce_lr_patience": 50,
    "closs_weight_schedule": "anneal",
    "c_lambda": 1.0,
    "d_lambda": 1.0,
}
tc = train_tabdiff(proc, train_params=train_params)

# TODO: need to figure out how to save/load model for inference.

# %%

syn_df = eval_tabdiff(proc, train_params=train_params)
