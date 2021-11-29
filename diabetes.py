import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib
warnings.filterwarnings("ignore")

class diabetes:
    def __init__(self):
        pass
    def readData(filename):
        data=pd.read_csv(filename)
        return data

    def preProcess(data,column):
        data=data.drop([column],axis=1)
        return data

    def fillMean(df,column):
        column=column
        df[column]=df[column].replace(to_replace=0,value=df[column].mean())
        return df

    def removeOutliers(df,column):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        q2=q3-q1
        df[column]=df[~((df[column]<(q1-1.5*q2)) | (df[column]>(q3+1.5*q2)))]
        df=df.dropna()
        return df


    def main(self):
        data = readData('diabetes.csv')
        data = preProcess(data, "Insulin")
        data = preProcess(data, "Pregnancies")
        data = fillMean(data, "Glucose")
        data = removeOutliers(data, "SkinThickness")
        x = data.drop(["Outcome"], axis=1)
        y = data["Outcome"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        rf = RandomForestClassifier(random_state=2)
        rf.fit(x_train, y_train)
        rfpred = rf.predict(x_test)
        print(accuracy_score(y_test, rfpred))
        joblib.dump(rf, "diabetesmodel.pkl")

        pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestClassifier(random_state=2))])
        pipe.fit(x_train, y_train)
        print(pipe.score(x_test, y_test))

    def predication(self):
        Glucose=130
        BloodPressure=70
        SkinThickness=148
        BMI=30
        DiabetesPedigreeFunction=1
        Age=22

        list = [[ Glucose, BloodPressure, SkinThickness, BMI,DiabetesPedigreeFunction, Age]]
        df = pd.DataFrame(list , columns=[ 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI','DiabetesPedigreeFunction', 'Age'])
        model=joblib.load("./models/diabetesmodel.pkl")
        pred = model.predict(df)
        if pred == 0 :
            print("There are no chance for diabetes")
        else :
            print("There is a possibility of diabetes we would request you to meet a cardiologist")

d = diabetes()
d.predication()