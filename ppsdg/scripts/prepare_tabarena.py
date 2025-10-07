# %%
import json
from pathlib import Path

import pandas as pd
from tabkit import DatasetConfig


def main():
    with open("tabarena-datasets/tabarena-datasets.json") as f:
        dsets = json.load(f)

    # %%

    config_files = []

    for dset in dsets:
        dset_name = "tabarena-" + dset["name"]
        config = DatasetConfig(
            config_name=dset_name,
            dataset_name=dset["name"],
            data_source="openml",
            openml_task_id=dset["task_id"],
            openml_split_idx=0,
        )

        config_file = f"config/dataset/{dset_name}.yaml"
        config.save_yaml(config_file)
        config_files.append(
            {
                "filename": config_file,
                "task_kind": dset["task_type"],
            }
        )

    save_dir = Path("tabarena-datasets/")
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(config_files)
    df.to_csv(save_dir / "metadata.csv", index=False)
    with open(save_dir / "clf.txt", "w") as f:
        f.write(
            "\n".join(df[df["task_kind"] == "classification"]["filename"].to_list())
        )

    with open(save_dir / "reg.txt", "w") as f:
        f.write("\n".join(df[df["task_kind"] == "regression"]["filename"].to_list()))


# %%
if __name__ == "__main__":
    main()
