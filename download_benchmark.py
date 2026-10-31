import urllib.request
import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
try:
    df = pd.read_csv(url, header=None, names=cols)
    df.to_csv("pima_diabetes_dataset.csv", index=False)
    print("Downloaded successfully. Shape:", df.shape)
    print(df.head(2))
except Exception as e:
    print("Download error:", e)
