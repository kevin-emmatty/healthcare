import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from glob import glob

class pneumonia:
  def __init__(self):
    pass
  def main(self):
    # re-size all the images to this
    IMAGE_SIZE = [224, 224]
    train_path = r'/content/drive/MyDrive/chest_xray/train'
    valid_path = '/content/drive/MyDrive/chest_xray/val'
    test_path =  r'/content/drive/MyDrive/chest_xray/test'

    # Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
    # Here we will be using imagenet weights
    vgg16 =  VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False) #[3] indicates the RGB channel
    # don't train existing weights
    for layer in vgg16.layers:
        layer.trainable = False
    # our layers - you can add more if you want
    x = Flatten()(vgg16.output)

    prediction = Dense(1, activation='sigmoid')(x)
    # create a model object
    model = Model(inputs=vgg16.input, outputs=prediction)

    # tell the model what cost and optimization method to use
    model.compile(
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )
    # Use the Image Data Generator to import the images from the dataset
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      rotation_range=30,
                                      width_shift_range=0.2,
                                      vertical_flip = True,
                                      horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)
    # Make sure you provide the same target size as initialied for the image size
    training_set = train_datagen.flow_from_directory(r'/content/drive/MyDrive/chest_xray/train',
                                                 target_size = (224, 224),
                                                 batch_size = 8,
                                                 class_mode = 'binary')
    validation_set = test_datagen.flow_from_directory(r'/content/drive/MyDrive/chest_xray/val',
                                            target_size = (224, 224),
                                            batch_size = 8,
                                            class_mode = 'binary')
    test_set = test_datagen.flow_from_directory(r'/content/drive/MyDrive/chest_xray/test',
                                            target_size = (224, 224),
                                            batch_size = 8,
                                            class_mode = 'binary')
    # fit the model
    # Run the cell. It will take some time to execute
    from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
    mc = ModelCheckpoint(monitor='val_accuracy',save_best_only=True,filepath='/content/drive/MyDrive/chest_xray',mode='max')
    es = EarlyStopping(patience=10)
    rlp = ReduceLROnPlateau(patience=5,monitor='val_accuracy',min_lr=0.001)
    r = model.fit_generator(training_set, validation_data=test_set, epochs= 50 ,validation_steps = len(test_set),callbacks=[es,rlp,mc])

  def prediction(self, images):
    from tensorflow import keras
    model = keras.models.load_model('./models/pneumonia.h5')
    import numpy as np
    from tensorflow.keras.preprocessing import image
    test_image = image.load_img(images, target_size = (112, 112))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'Normal'
        print(prediction)
    elif result[0][1]:
        prediction = 'Positive'
        print(prediction)

p= pneumonia()
p.prediction("IM-0119-0001.jpeg")