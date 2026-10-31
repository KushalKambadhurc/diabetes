import pandas as pd
import numpy as np

print("=== PRIMARY DATASET: diabetes_prediction_dataset.csv ===")
df_main = pd.read_csv("diabetes_prediction_dataset.csv")
print("Shape:", df_main.shape)
print("Null values:\n", df_main.isnull().sum())
print("\nData Types:\n", df_main.dtypes)
print("\nTarget Distribution:\n", df_main['diabetes'].value_counts(normalize=True))
print("\nGender value counts:\n", df_main['gender'].value_counts())
print("\nSmoking history value counts:\n", df_main['smoking_history'].value_counts())
print("\nNumerical stats:\n", df_main.describe())

print("\n=== BENCHMARK DATASET: pima_diabetes_dataset.csv ===")
df_pima = pd.read_csv("pima_diabetes_dataset.csv")
print("Shape:", df_pima.shape)
print("Null values:\n", df_pima.isnull().sum())
print("\nTarget Distribution:\n", df_pima['outcome'].value_counts(normalize=True))
print("\nNumerical stats:\n", df_pima.describe())
