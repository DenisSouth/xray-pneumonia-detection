# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from google.colab import files

# Prepare dataset
# download https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
# after unzip - you got 3 folder  with pics
# content/chest_xray/test
# content/chest_xray/train
# content/chest_xray/val

# Make model
base_model = MobileNet(weights='imagenet',
                       include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(
    x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dense(1024, activation='relu')(x)  # dense layer 2
x = Dense(512, activation='relu')(x)  # dense layer 3
preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

# Train model"""

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('/content/chest_xray/train/',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical', shuffle=True)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n // train_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=5)

# Save the weights

model.save('chest-xray-pneumonia.h5')

# Implement

from keras.models import load_model

new_model = load_model("/content/chest-xray-pneumonia.h5")


def get_rez(pic):
    img = image.load_img(pic, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    p_good, p_ill = np.around(new_model.predict(x), decimals=2)[0]
    return {'p_good': p_good, 'p_ill': p_ill}


ill_path = "/content/chest_xray/train/PNEUMONIA/"
good_path = "/content/chest_xray/train/NORMAL/"

ill_pic = ill_path + os.listdir(ill_path)[0]
good_pic = good_path + os.listdir(good_path)[5]

print(get_rez(ill_pic))
print(get_rez(good_pic))
