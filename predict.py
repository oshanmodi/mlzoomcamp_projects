import pickle
import pandas as pd
from utils import path_model, path_test_csv

with open(path_model, "rb") as f_in:
    dv, model = pickle.load(f_in)

dftest = pd.read_csv(path_test_csv)
dftest.column = [col.lower() for col in dftest.columns]

cols_categorical = ["pclass", "sex", "embarked"]
cols_numeric = ["age", "sibsp", "parch", "fare"]
col_target = "survived"

dftest_no_missing = dftest[cols_categorical + cols_numeric].ffill(axis=1).bfill(axis=1)
dftest_dict = dftest_no_missing.to_dict(orient="records")
xtest = dv.transform(dftest_dict)

test_predictions = model.predict_proba(xtest)[:, 1]

with open("test_predictions.csv", "w") as fout:
    for val in test_predictions:
        fout.writelines(val.astype(str) + "\n")
