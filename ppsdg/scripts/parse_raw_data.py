# %%

from pathlib import Path

import pandas as pd

data_dir = Path("/Users/inwon/Downloads/ppsdg-data")


# %%

"""
Lending Club
... this is not transactional data!
"""

rej = pd.read_csv(data_dir / "lendingclub/rejected_2007_to_2018Q4.csv")
acc = pd.read_csv(data_dir / "lendingclub/accepted_2007_to_2018Q4.csv")

# skip everything, just use the accepted data
acc.to_csv("lendingclub.csv", index=False)

# %%

"""
SWIFT data
"""

tr = pd.read_csv(data_dir / "swift/swift_transaction_train_dataset.csv")
te = pd.read_csv(data_dir / "swift/swift_transaction_test_dataset.csv")
ba = pd.read_csv(data_dir / "swift/bank_dataset.csv")

df = (
    pd.concat([tr, te])
    .merge(
        ba,
        left_on="BeneficiaryAccount",
        right_on="Account",
        how="inner",
    )
    .drop(
        columns=["Account"],
    )
)

df.to_csv("swift.csv", index=False)

# %%

"""
FAR-Trans
"""

assets = pd.read_csv(data_dir / "far-trans/asset_information.csv")
close_prices = pd.read_csv(data_dir / "far-trans/close_prices.csv")
customers = pd.read_csv(data_dir / "far-trans/customer_information.csv")
limit_prices = pd.read_csv(data_dir / "far-trans/limit_prices.csv")
markets = pd.read_csv(data_dir / "far-trans/markets.csv")
transactions = pd.read_csv(data_dir / "far-trans/transactions.csv")

"""
The customers table has duplicates, b/c customer info is updated over time. 
Need to join the correct customer "state" for the transaction.
(i.e. we can't have the future version of the customer info for a transaction)
"""

transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
customers["timestamp"] = pd.to_datetime(customers["timestamp"])
assets["timestamp"] = pd.to_datetime(assets["timestamp"])
close_prices["timestamp"] = pd.to_datetime(close_prices["timestamp"])
limit_prices["minDate"] = pd.to_datetime(limit_prices["minDate"])
limit_prices["maxDate"] = pd.to_datetime(limit_prices["maxDate"])

transactions = transactions.sort_values(by=["timestamp"])
customers = customers.sort_values(by=["timestamp"])
assets = assets.sort_values(by=["timestamp"])
close_prices = close_prices.sort_values(by=["timestamp"])
limit_prices = limit_prices.sort_values(by=["maxDate"])

# %%

trans_w_market = transactions.merge(markets, on="marketID")

# first join the most up-to-date customer info
merged = pd.merge_asof(
    left=trans_w_market,
    right=customers,
    on="timestamp",
    by="customerID",
    direction="backward",
)

# next join the most up-to-date asset info
merged2 = pd.merge_asof(
    left=merged,
    right=assets,
    on="timestamp",
    by="ISIN",
    direction="backward",
)

# join the last day's closing prices
# we can't have the closing price for the same day as the transaction,
# so we need to shift the timestamp by one day
close_prices["timestamp"] = close_prices["timestamp"] + pd.DateOffset(days=1)
merged3 = pd.merge_asof(
    left=merged2,
    right=close_prices,
    on="timestamp",
    by="ISIN",
    direction="backward",
)

# finally, join the limit price analysis
# but ONLY if the "max_day" is **before** the transaction date
# we will use the timestamp column as the dummy key to merge on

limit_prices["timestamp"] = limit_prices["maxDate"] + pd.DateOffset(days=1)
merged4 = pd.merge_asof(
    left=merged3,
    right=limit_prices,
    on="timestamp",
    direction="backward",
)
merged4.to_csv("far-trans.csv", index=False)

# %%

"""
convert the AAVE rds data to python-friendly format
"""

import pyreadr

df = pyreadr.read_r("/Users/inwon/Downloads/ppsdg-data/aave/transactions.rds")[None]
uu = pyreadr.read_r("/Users/inwon/Downloads/ppsdg-data/aave/transactions_user.rds")[
    None
]


merged = (
    df.merge(users, left_on="user", right_on="id")
    .drop(columns=["version_x", "version_y", "deployment_x", "deployment_y", "id_y"])
    .rename(columns={"id_x": "id"})
)
