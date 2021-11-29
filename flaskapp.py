from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input , decode_predictions
import numpy as np
import os

def get_model():
    global model
    model = load_model('./models/braintumour.h5')
    print("Model loaded!")

def load_image(img_path):

    img = image.load_img(img_path, target_size=(112, 112))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor


def prediction(img_path):
	new_image = load_image(img_path)

	pred = model.predict(new_image)

	print(pred)

	labels = np.array(pred)
	labels[labels >= 0.6] = 1
	labels[labels < 0.6] = 0

	print(labels)
	final = np.array(labels)

	if final[0][0] == 1:
		return "No Tumour"
	else:
		return "Have Tumour"

app = Flask(__name__)

get_model()

@app.route("/", methods=['GET', 'POST'])
def home():
	return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		file = request.files['file']
		filename = file.filename
		file_path = os.path.join(r'./static/', filename)  # slashes should be handeled properly
		file.save(file_path)
		print(filename)
		product = prediction(file_path)
		print(product)

	return render_template('predict.html', product=product, user_image=file_path)

if __name__ == '__main__':
	app.run(debug = True )