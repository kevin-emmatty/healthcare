from PIL.Image import Image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

scaler = StandardScaler()

app = Flask(__name__)
model = joblib.load('./models/heartattack.pkl')
@app.route("/", methods = ['GET'])
def renderer():
    return render_template("index.html")

@app.route("/main", methods=['POST'])
def select_model():
    model=request.form['pred_model']
    if(model=="heart"):
        return render_template('heartAttack.html')
    if(model=="diabetes"):
        return render_template('diabetes.html')
    if(model=="brain"):
        return render_template('brain_tumor.html')
    if(model=="lung"):
        return render_template('pneumonia.html')


@app.route("/menu", methods=['GET','POST'])
def main_menu():
    return render_template('index.html')

@app.route("/heart_prediction", methods=['POST'])
def pred_ha():
    age = request.form['age']
    sex = request.form['sex']
    if sex == "Male":
        sex = 1
    else:
        sex = 0
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = float(request.form['oldpeak'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])

    list = [[age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak, ca, thal]]
    df = pd.DataFrame(list,
                      columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang', 'oldpeak',
                               'ca', 'thal'])
    mms = MinMaxScaler()
    df=mms.fit_transform(df)
    model = joblib.load("./models/heartattack.pkl")
    result = model.predict(df)
    if result == 1:
        return render_template("heartAttack.html",pred_text='You have chances of getting heart attack')
    else:

        return render_template("heartAttack.html",pred_text='You have less chances of getting heart attack')

@app.route("/diabetes_prediction", methods=['POST'])
def pred_diabetes():
    Glucose = request.form['Glucose']
    BloodPressure = request.form['BloodPressure']
    SkinThickness = request.form['SkinThickness']
    BMI = request.form['BMI']
    DiabetesPedigreeFunction =request.form['DiabetesPedigreeFunction']
    Age = int(request.form['Age'])

    list = [[Glucose, BloodPressure, SkinThickness, BMI, DiabetesPedigreeFunction, Age]]
    df = pd.DataFrame(list,
                      columns=['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    mms = MinMaxScaler()
    df = mms.fit_transform(df)
    model = joblib.load("./models/diabetesmodel.pkl")
    result = model.predict(df)
    if result == 1:
        return render_template("diabetes.html",pred_text='You have chances of getting diabetes')
    else:
        return render_template("diabetes.html",pred_text='You don\'t have chances of getting diabetes')

########################################################################################################################

def brain_get_model():
    global model
    model = load_model('./models/braintumour.h5')
    print("Model loaded!")

def brain_load_image(img_path):

    img = image.load_img(img_path, target_size=(112, 112))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor


def brain_prediction(img_path):
    new_image = brain_load_image(img_path)
    pred = model.predict(new_image)
    labels = np.array(pred)
    labels[labels >= 0.6] = 1
    labels[labels < 0.6] = 0

    print(labels)
    final = np.array(labels)

    if final[0][0] == 1:
        return "No Tumour"
    else:
        return "Have Tumour"

brain_get_model()
#
# @app.route("/brain", methods=['GET', 'POST'])
# def _brain():
# 	return render_template('brain_tumor.html')


@app.route("/predict_tumor", methods=['GET', 'POST'])
def pred_tumor():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'./static/', filename)  # slashes should be handeled properly
        file.save(file_path)
        print(filename)
        product = brain_prediction(file_path)
        print(product)

    return render_template('predict_tumor.html', product=product, user_image=file_path)

########################################################################################################################

def pneumonia_get_model():
    global model
    model = load_model('./models/pneumonia.h5')
    # print("Model loaded!")

def pneumonia_load_image(img_path):

    img = image.load_img(img_path, target_size=(112, 112))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor


def pneumonia_prediction(img_path):
	new_image = pneumonia_load_image(img_path)
	pred = model.predict(new_image)
	labels = np.array(pred)
	labels[labels >= 0.6] = 1
	labels[labels < 0.6] = 0
	print(labels)
	final = np.array(labels)
	if final[0][0] == 1:
		return "Chances of Pneumonia are close to zero"
	else:
		return "Chances of Pneumonia are high"

pneumonia_get_model()
#
# @app.route("/pneumonia", methods=['GET', 'POST'])
# def home():
# 	return render_template('pneumonia.html')


@app.route("/predict_pneumonia", methods=['GET', 'POST'])
def pred_pneumonia():
	if request.method == 'POST':
		file = request.files['file']
		filename = file.filename
		file_path = os.path.join(r'./static/', filename)  # slashes should be handeled properly
		file.save(file_path)
		product = pneumonia_prediction(file_path)

	return render_template('predict_pneumonia.html', product=product, user_image=file_path)

if __name__ == "__main__":
    app.run(debug=True)