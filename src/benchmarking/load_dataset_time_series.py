from datasets import load_dataset
import numpy as np
import pandas as pd

# load the Australian electricity demand dataset
australian_electricity_demand_ds = load_dataset("Monash-University/monash_tsf", "australian_electricity_demand", trust_remote_code=True)
solar_4_seconds_ds = load_dataset("Monash-University/monash_tsf", "solar_4_seconds", trust_remote_code=True)
wind_4_seconds_ds = load_dataset("Monash-University/monash_tsf", "wind_4_seconds", trust_remote_code=True)
oikolab_weather_ds = load_dataset("Monash-University/monash_tsf", "oikolab_weather", trust_remote_code=True)

# print(australian_electricity_demand_ds)  # shows available splits (train/validation/test)

# extract all target sequences 
australian_electricity_demand_targets = [np.array(ex["target"]) for ex in australian_electricity_demand_ds["train"]]  # time series
pd.DataFrame(australian_electricity_demand_targets[0]).to_csv("loaded_time_series/australian_electricity_demand_series_0.csv", index=False, header=False)

solar_4_seconds_targets = [np.array(ex["target"]) for ex in solar_4_seconds_ds["train"]]  # time series
pd.DataFrame(solar_4_seconds_targets[0]).to_csv("loaded_time_series/solar_4_seconds_series_0.csv", index=False, header=False)

wind_4_seconds_targets = [np.array(ex["target"]) for ex in wind_4_seconds_ds["train"]]  # time series
pd.DataFrame(wind_4_seconds_targets[0]).to_csv("loaded_time_series/wind_4_seconds_series_0.csv", index=False, header=False)

oikolab_weather_targets = [np.array(ex["target"]) for ex in oikolab_weather_ds["train"]]  # time series
pd.DataFrame(oikolab_weather_targets[0]).to_csv("loaded_time_series/oikolab_weather_series_0.csv", index=False, header=False)

# print some information
# print(f"Number of series: {len(australian_electricity_demand_targets)}")
# print(f"Length of first series: {len(australian_electricity_demand_targets[0])}")
# print("First series (first 10 values):", australian_electricity_demand_targets[0][:10])

# print(f"Number of series: {len(solar_4_seconds_targets)}")
# print(f"Length of first series: {len(solar_4_seconds_targets[0])}")
# print("First series (first 10 values):", solar_4_seconds_targets[0][:10])

# print(f"Number of series: {len(wind_4_seconds_targets)}")
# print(f"Length of first series: {len(wind_4_seconds_targets[0])}")
# print("First series (first 10 values):", wind_4_seconds_targets[0][:10])

# print(f"Number of series: {len(oikolab_weather_targets)}")
# print(f"Length of first series: {len(oikolab_weather_targets[0])}")
# print("First series (first 10 values):", oikolab_weather_targets[0][:10])