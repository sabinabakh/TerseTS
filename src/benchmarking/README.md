# Use Monasch Dataset

## Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate 

pip install "datasets==2.19.0" "huggingface_hub<0.25" #newer datasets>=4.0.0 removed support for script loaders, so we use an older version
pip install numpy pandas
```

## Pick a dataset “config” (subset)
```py
from datasets import get_dataset_config_names

configs = get_dataset_config_names("Monash-University/monash_tsf")

print(len(configs), configs[:15])  # sample
```

## Load Dataset 
```py 
from datasets import load_dataset

# load the NN5 daily dataset
ds = load_dataset("Monash-University/monash_tsf", "nn5_daily", trust_remote_code=True)

print(ds)  # shows available splits (train/validation/test)
```
## Extract target values
```py 
import numpy as np

train_split = ds["train"]

# extract all target sequences
targets = [np.array(ex["target"]) for ex in train_split]

print(f"Number of series: {len(targets)}")
print(f"Length of first series: {len(targets[0])}")
print("First series (first 10 values):", targets[0][:10])
```

## Save as CSV files:
```py
import pandas as pd

# export the first series
pd.DataFrame(targets[0]).to_csv("series_0.csv", index=False)
```

# Run the Benchmark

# Build TerseTS 
```bash
cd /TerseTS
zig build
```
## Navigate to the benchmarking directory
```bash
cd integer_encoding_benchmarking
```

## Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate 

pip install "datasets==2.19.0" "huggingface_hub<0.25" #newer datasets>=4.0.0 removed support for script loaders, so we use an older version
pip install numpy pandas
```

# Load the time Series
```py
python load_dataset_time_series #loads the datasets and saves teh time series to loaded_time_series folder
```

# Run the benchmark script
```bash
# From the project root
cd /TerseTS
zig run src/integer_encoding_benchmark.zig
```