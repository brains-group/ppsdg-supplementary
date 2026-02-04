# Measuring Privacy Risks and Tradeoffs in Financial Synthetic Data Generation

This is the official codebase for the paper titled "Measuring Privacy Risks and Tradeoffs in Financial Synthetic Data Generation".
Both the model implementations as well as the evaluation configuration/scripts are included in this repository.

## Getting Started

1. Create a virtual environment using conda:

```bash
conda create -n ppsdg python=3.10 -y
```

2. Install the project as a package:

```bash
pip install -e .[dev]
```

(the `[dev]` part is optional, it will install additional development dependencies)

Now any script that contains `__main__` can be executed like this:

```bash
python -m ppsdg.<script_dir>.<script_name>
```

Just replace `/` with `.` and remove the `.py` extension.

3. Create a `.env` file in the root directory of the project with the following
   content. Refer to the `.env.example` file for more details:

Note: When debugging, make sure to include `import ppsdg` at the top level so
that the environment variables are loaded correctly.

## Datasets

For the configurations to work correctly, the dataset files should be placed in
`/home/shared/ppsdg-raw-data` directory. Or, the configurations can be modified
to point to the correct dataset paths. Keep in mind that changing the
configuration will change the hash of that particular configuration (for
reproducibility purposes).

The datasets can be loaded using [tabkit](https://github.com/inwonakng/tabkit)
package, which will take care of basic column type inference and preprocessing
(if necessary), like the following:

```python
from tabkit import DatasetConfig, TableProcessorConfig, TableProcessor

processor = TableProcessor(
    dataset_config = DatasetConfig.from_yaml("config/dataset/bank_marketing.yaml"),
    config = TableProcessorConfig(), # use default
).prepare() # this needs to be called to cache/load everything.

```

### Training synthesizer/generating data

This is handled by `ppsdg.evaluate.train_and_generate` script. If this
repository is installed as a package, the `train-gen` command is available. It
takes two configurations, one for the synthesizer and one for actually
generating the synthetic data.
