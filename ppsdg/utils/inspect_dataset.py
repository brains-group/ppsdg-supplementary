import argparse
import json

from tabkit.data import DatasetConfig, TableProcessor, TableProcessorConfig


def split_special(name):
    # Split by numbers and underscores, and CamelCase
    parts = re.findall(r"[A-Za-z]+|\d+", name)
    camel_split = []
    for part in parts:
        # Split camel case segments separately and extend to the list
        camel_split.extend(re.findall(r"[a-z]+|[A-Z][a-z]*", part))
    return camel_split


def inspect_dataset(dataset_config: str):
    config = DatasetConfig.from_yaml(dataset_config)
    proc = TableProcessor(
        dataset_config=config,
        config=TableProcessorConfig(),
    ).prepare()
    X_tr, y_tr = proc.get_split(split="train")
    X_va, y_va = proc.get_split(split="validation")
    X_te, y_te = proc.get_split(split="test")
    print()
    print("## {} Metadata".format(config.config_name))
    print()
    print("**Task type**: {}".format(proc.label_info.kind))
    print("{} rows, {} columns".format(proc.n_samples, proc.n_cols))
    n_cat = len([c for c in proc.columns_info if c.kind == "categorical"])
    print("{} cat, {} cont".format(n_cat, len(proc.columns_info) - n_cat))
    print()
    print("----")
    for col in proc.columns_info:
        print("**{}**".format(col.name))
        print("  - kind: {}".format(col.kind))
        print("  - dtype: {}".format(col.dtype))
        stats = None
        if col.is_num:
            stats = {
                "min": {
                    "train": float(X_tr[col.name].min()),
                    "validation": float(X_va[col.name].min()),
                    "test": float(X_te[col.name].min()),
                },
                "max": {
                    "train": float(X_tr[col.name].max()),
                    "validation": float(X_va[col.name].max()),
                    "test": float(X_te[col.name].max()),
                },
                "mean": {
                    "train": float(X_tr[col.name].mean()),
                    "validation": float(X_va[col.name].mean()),
                    "test": float(X_te[col.name].mean()),
                },
                "median": {
                    "train": float(X_tr[col.name].median()),
                    "validation": float(X_va[col.name].median()),
                    "test": float(X_te[col.name].median()),
                },
            }
        if col.is_cat:
            stats = {
                "n_unique": {
                    "train": int(X_tr[col.name].nunique()),
                    "validation": int(X_va[col.name].nunique()),
                    "test": int(X_te[col.name].nunique()),
                },
                "top_value": {
                    "train": int(X_tr[col.name].mode()[0]),
                    "validation": int(X_va[col.name].mode()[0]),
                    "test": int(X_te[col.name].mode()[0]),
                },
                "top_count": {
                    "train": int(X_tr[col.name].value_counts().max()),
                    "validation": int(X_va[col.name].value_counts().max()),
                    "test": int(X_te[col.name].value_counts().max()),
                },
            }
        if stats:
            print()
            print("  - stats:")
            print(json.dumps(stats, indent=2))
            print()
        if col.mapping:
            print("  - sample values:")
            for v in col.mapping[:5]:
                print("    - {}".format(v))
            print("  - {} total values".format(len(col.mapping)))
        print()
    print("----")
    print("**{}**".format(proc.label_info.name))
    print("  - kind: {}".format(proc.label_info.kind))
    print("  - dtype: {}".format(proc.label_info.dtype))
    if proc.label_info.mapping:
        _, y = proc.get_split()
        print("  - sample values:")
        for k, v in enumerate(proc.label_info.mapping[:5]):
            print("    - {}: {} [{}]".format(k, v, (y == int(k)).sum()))
        print("  - {} total values".format(len(proc.label_info.mapping)))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, help="name of the dataset to inspect")
    args = parser.parse_args()
    inspect_dataset(dataset_config=args.dataset_name)


if __name__ == "__main__":
    main()
