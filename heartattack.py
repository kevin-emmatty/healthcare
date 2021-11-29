import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib


class heartAttack:
    def __init__(self, df, scaler, model):
        self.df = pd.read_csv(df)
        self.scaler = scaler
        self.model = model

    # main function for calling model
    def main(self):
        data = self.df
        x = data.drop(["target", "fbs", "slope"], axis=1)
        y = data["target"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        pipe = Pipeline([('scaler', self.scaler), ('gnb', self.model)])
        pipe.fit(x_train, y_train)
        print(pipe.score(x_test, y_test))

    def predict(self):
        age = 22
        sex = 1
        cp = 3
        trestbps = 120
        chol = 250
        restecg = 0
        thalach = 187
        exang = 0
        oldpeak = 3.5
        ca = 0
        thal = 2

        list = [[age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak, ca, thal]]
        df = pd.DataFrame(list,
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang', 'oldpeak',
                                   'ca', 'thal'])
        df=MinMaxScaler(df)
        model = joblib.load("./models/heartattack.pkl")
        result = model.predict(df)
        if result == 1:
            print("you have heart issues")
        else:
            print("you dont have heart issues")

ha = heartAttack("heart.csv", MinMaxScaler(), GaussianNB())
ha.predict()
