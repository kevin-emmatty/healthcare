import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob


class brainTumour:
    def __init__(self):
        pass

    def main(self):
        # re-size all the images to this
        IMAGE_SIZE = [112, 112]

        train_path = r'\content\drive\MyDrive\model\brainTumour\brain_tumor_dataset'
        valid_path = r'\content\drive\MyDrive\model\brainTumour\brain_tumor_dataset'
        # Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
        # Here we will be using imagenet weights
        vgg16 = VGG16(input_shape=(112, 112, 3), weights='imagenet', include_top=False)  # [3] indicates the RGB channel

        # don't train existing weights
        for layer in vgg16.layers:
            layer.trainable = False

        # our layers - you can add more if you want
        x = Flatten()(vgg16.output)
        prediction = Dense(len(folders), activation='softmax')(x)

        # create a model object
        model = Model(inputs=vgg16.input, outputs=prediction)

        # tell the model what cost and optimization method to use
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # Use the Image Data Generator to import the images from the dataset
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Make sure you provide the same target size as initialied for the image size
        training_set = train_datagen.flow_from_directory(
            r'\content\drive\MyDrive\model\brainTumour\brain_tumor_dataset',
            target_size=(112, 112),
            batch_size=16,
            class_mode='categorical')
        test_set = test_datagen.flow_from_directory(r'\content\drive\MyDrive\model\brainTumour\brain_tumor_dataset',
                                                    target_size=(112, 112),
                                                    batch_size=16,
                                                    class_mode='categorical')

        # fit the model
        # Run the cell. It will take some time to execute
        r = model.fit_generator(training_set, validation_data=test_set, epochs=50, validation_steps=len(test_set))

    def predict(self, img):
        from tensorflow.keras.preprocessing import image
        model = keras.models.load_model('./models/braintumour.h5')
        test_image = image.load_img(img, target_size=(112, 112))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        if result[0][0] == 1:
            prediction = 'Not having tumour '
            print(prediction)
        elif result[0][1]:
            prediction = 'Yes u have chances of having tumour'
            print(prediction)


bt = brainTumour()
bt.predict('Y17.jpg')