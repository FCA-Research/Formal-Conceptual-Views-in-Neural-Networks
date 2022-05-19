import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Lambda, GlobalMaxPooling2D, MaxPooling1D

learning_rate = 0.1  # initial learning rate
min_learning_rate = 0.00001  # once the learning rate reaches this value, do not decrease it further
learning_rate_reduction_factor = 0.5  # the factor used when reducing the learning rate -> learning_rate *= learning_rate_reduction_factor
patience = 3  # how many epochs to wait before reducing the learning rate when the loss plateaus
verbose = 1  # controls the amount of logging done during training and testing: 0 - none, 1 - reports metrics after each batch, 2 - reports metrics after each epoch
image_size = (100, 100)  # width and height of the used images
input_shape = (100, 100, 3)  # the expected input shape for the trained models; since the images in the Fruit-360 are 100 x 100 RGB images, this is the required input shape

base_dir = 'image-data/fruit360'  # relative path to the Fruit-Images-Dataset folder
test_dir = os.path.join(base_dir, 'Test')
train_dir = os.path.join(base_dir, 'Training')
output_dir = 'fruit_models'  # root folder in which to save the the output files; the files will be under output_files/model_name

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

labels = os.listdir(train_dir)
num_classes = len(labels)

def build_data_generators(train_folder, test_folder, validation_percent, labels=None, image_size=(100, 100), batch_size=50):
    train_datagen = ImageDataGenerator(
        width_shift_range=0.0,
        height_shift_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,  # randomly flip images
        preprocessing_function=augment_image, 
        validation_split=validation_percent)  # percentage indicating how much of the training set should be kept for validation

    test_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(train_folder, target_size=image_size, class_mode='sparse',
                                                  batch_size=batch_size, shuffle=True, subset='training', classes=labels)
    validation_gen = train_datagen.flow_from_directory(train_folder, target_size=image_size, class_mode='sparse',
                                                       batch_size=batch_size, shuffle=False, subset='validation', classes=labels)
    test_gen = test_datagen.flow_from_directory(test_folder, target_size=image_size, class_mode='sparse',
                                                batch_size=batch_size, shuffle=False, subset=None, classes=labels)
    return train_gen, validation_gen, test_gen

def train_and_evaluate_model(model, name="", epochs=25, batch_size=50, verbose=verbose, useCkpt=False):
    print(model.summary())
    model_out_dir = os.path.join(output_dir, name)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    if useCkpt:
        model.load_weights(model_out_dir + "/model.h5")

    trainGen, validationGen, testGen = build_data_generators(train_dir, test_dir, validation_percent=0.1, labels=labels, image_size=image_size, batch_size=batch_size)
    optimizer = Adadelta(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=patience, verbose=verbose, 
                                                factor=learning_rate_reduction_factor, min_lr=min_learning_rate)
    save_model = ModelCheckpoint(filepath=model_out_dir + "/model.h5", monitor='val_accuracy', verbose=verbose, 
                                 save_best_only=True, save_weights_only=False, mode='max', period=1)

    history = model.fit(trainGen,
                                  epochs=epochs,
                                  steps_per_epoch=(trainGen.n // batch_size) + 1,
                                  validation_data=validationGen,
                                  validation_steps=(validationGen.n // batch_size) + 1,
                                  verbose=verbose,
                                  callbacks=[learning_rate_reduction, save_model])

    model.load_weights(model_out_dir + "/model.h5")

    validationGen.reset()
    loss_v, accuracy_v = model.evaluate(validationGen, steps=(validationGen.n // batch_size) + 1, verbose=verbose)
    loss, accuracy = model.evaluate(testGen, steps=(testGen.n // batch_size) + 1, verbose=verbose)
    print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
    print("Test: accuracy = %f  ;  loss_v = %f" % (accuracy, loss))

    testGen.reset()
    y_pred = model.predict(testGen, steps=(testGen.n // batch_size) + 1, verbose=verbose)
    y_true = testGen.classes[testGen.index_array]

    w, = model.layers[-1].trainable_weights
    pd.DataFrame(w.numpy().T,index=labels).to_csv(model_out_dir + "/weights.csv")

def evaluate_model(model,batch_size=50, verbose=verbose):
    optimizer = Adadelta(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    trainGen, validationGen, testGen = build_data_generators(train_dir, test_dir, validation_percent=0.1, labels=labels, image_size=image_size, batch_size=batch_size)
    loss, accuracy = model.evaluate(testGen, steps=(testGen.n // batch_size) + 1, verbose=verbose)

def augment_image(x):
    x = tf.image.random_saturation(x, 0.9, 1.2)
    x = tf.image.random_hue(x, 0.02)
    return x

def convert_to_hsv_and_grayscale(x):
    hsv = tf.image.rgb_to_hsv(x)
    gray = tf.image.rgb_to_grayscale(x)
    rez = tf.concat([hsv, gray], axis=-1)
    return rez

def network(input_shape, num_classes, activation="tanh",neurons_second_last=8,neurons_last=5):
    img_input = Input(shape=input_shape, name='data')
    x = Lambda(convert_to_hsv_and_grayscale)(img_input)
    x = Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='conv1')(x)
    x = Activation(activation, name='conv1_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool1')(x)
    x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv2')(x)
    x = Activation(activation, name='conv2_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool2')(x)
    x = Conv2D(64, (5, 5), strides=(1, 1), padding='same', name='conv3')(x)
    x = Activation(activation, name='conv3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool3')(x)
    x = Conv2D(128, (5, 5), strides=(1, 1), padding='same', name='conv4')(x)
    x = Activation(activation, name='conv4_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='pool4')(x)
    x = Flatten()(x)
    x = Dense(1024, activation=activation, name='fcl1')(x)
    x = Dropout(0.2)(x)
    x = Dense(2**neurons_second_last, activation=activation, name='fcl2')(x)
    x = Dropout(0.2)(x)
    x = Dense(2**neurons_last, activation=activation, name='fcl3')(x) #32
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation='softmax', name='predictions',use_bias=False)(x)
    rez = Model(inputs=img_input, outputs=out)
    return rez

def model_to_16(model,num_classes):
    xi16 = Dense(16, activation='tanh')(model.layers[-2].output)
    xi16 = Dense(num_classes, activation='softmax',use_bias=False)(xi16)
    model16 = Model(model.input, xi16)
    return model16

def network16(input_shape,num_classes):
    model = network(input_shape=input_shape, num_classes=num_classes)
    return model_to_16(model,num_classes)

base = network(input_shape=input_shape, num_classes=num_classes)

def transfer_learn_model(model):
    model.trainable = False

    xv = Flatten()(pretrained_model_vgg.layers[-1].output)
    xv = Dense(1024, activation='tanh', name='tan1')(xv)
    xv = Dropout(0.2)(xv)
    xv = Dense(256, activation='tanh', name='tan2')(xv)
    xv = Dropout(0.2)(xv)
    xv = Dense(32, activation='tanh', name='tan3')(xv)
    xv = Dense(num_classes, activation='softmax',use_bias=False,name="out")(xv)

    return Model(pretrained_model_vgg.input, xv)

from tensorflow.keras.applications import VGG16
vgg16 = VGG16(input_shape=(input_shape[0], input_shape[1], input_shape[2]), include_top=False, weights="imagenet")
transfer_learn_model(vgg16)

from tensorflow.keras.applications.resnet50 import ResNet50
resnet50 = ResNet50(input_shape=(input_shape[0], input_shape[1], input_shape[2]), include_top=False, weights="imagenet")
transfer_learn_model(resnet50)

from tensorflow.keras.applications.inception_v3 import InceptionV3
incv3 = InceptionV3(input_shape=(input_shape[0], input_shape[1], input_shape[2]), include_top=False, weights="imagenet")
transfer_learn_model(incv3)

from tensorflow.keras.applications.efficientnet import EfficientNetB0
effb0 = EfficientNetB0(input_shape=(input_shape[0], input_shape[1], input_shape[2]), include_top=False, weights="imagenet")
transfer_learn_model(effb0)

def train_transfer_learn_model(model,name,epochs_freeze=5,epochs_all=5):
    learning_rate=0.1
    train_and_evaluate_model(model, name=name,epochs=epochs_freeze)
    learning_rate=0.05
    model.trainable=True
    train_and_evaluate_model(model, name=name,epochs=all)

train_and_evaluate_model(base, name="base-32-tanh",epochs=10)
train_transfer_learn_model(vgg16,"vgg16-32-tanh",epochs_freeze=5,epochs_all=5)
train_transfer_learn_model(resnet50,"resnet50-32-tanh",epochs_freeze=2,epochs_all=3)
train_transfer_learn_model(incv3,"incv3-32-tanh",epochs_freeze=5,epochs_all=5)
train_transfer_learn_model(effb0,"effb0-32-tanh",epochs_freeze=3,epochs_all=3)

def transfer_to_16(model,name,epochs_freeze=5,epochs_all=5):
    learning_rate=0.1
    model.trainable = False
    model16 = model_to_16(model,num_classes)
    train_and_evaluate_model(model16, name=name,epochs=epochs_freeze)
    model.trainable = True
    train_and_evaluate_model(model16, name=name,epochs=epochs_all)

transfer_to_16("base-32-tanh",epochs_freeze=5,epochs_all=10)
transfer_to_16("vgg16-32-tanh",epochs_freeze=5,epochs_all=5)
transfer_to_16("resnet50-32-tanh",epochs_freeze=5,epochs_all=5)
transfer_to_16("incv3-32-tanh",epochs_freeze=3,epochs_all=5)
transfer_to_16("effb0-32-tanh",epochs_freeze=5,epochs_all=5)
