import pandas as pd

train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

# create row hash
train["hash"] = train.apply(lambda x: hash(tuple(x)), axis=1)
test["hash"] = test.apply(lambda x: hash(tuple(x)), axis=1)

# remove overlap from test
test = test[~test["hash"].isin(train["hash"])]

print("Remaining overlap:",
      len(set(train["hash"]).intersection(set(test["hash"]))))

# drop helper column
train.drop(columns=["hash"], inplace=True)
test.drop(columns=["hash"], inplace=True)

# save cleaned files
train.to_csv("../data/processed/train.csv", index=False)
test.to_csv("../data/processed/test.csv", index=False)

print("Overlap fixed and files saved.")