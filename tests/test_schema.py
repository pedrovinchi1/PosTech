import pandas as pd
import pandera as pa
import pytest

schema = pa.DataFrameSchema({
    "tenure": pa.Column(int, pa.Check.ge(0)),
    "MonthlyCharges": pa.Column(float, pa.Check.ge(0)),
    "TotalCharges": pa.Column(float, nullable=True),
    "Churn": pa.Column(str, pa.Check.isin(["Yes", "No"])),
})


def test_schema_valid():
    df = pd.DataFrame({
        "tenure": [12, 0],
        "MonthlyCharges": [65.5, 29.85],
        "TotalCharges": [786.0, None],
        "Churn": ["No", "Yes"],
    })
    schema.validate(df)


def test_schema_rejects_negative_tenure():
    df = pd.DataFrame({
        "tenure": [-1],
        "MonthlyCharges": [50.0],
        "TotalCharges": [600.0],
        "Churn": ["No"],
    })
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)


def test_real_dataset_schema():
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    schema.validate(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])
