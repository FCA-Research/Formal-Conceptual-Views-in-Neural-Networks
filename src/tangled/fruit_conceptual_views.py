import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model

if not os.path.exists("scales/fruit_class"):
    os.makedirs("scales/fruit_class")
if not os.path.exists("scales/fruit_obj"):
    os.makedirs("scales/fruit_obj")

import numpy as np
import pandas as pd

def make_scale(w,delta=np.zeros(4096,)):
     """The many-valued scale w and a delta array that specifies the
     scaling thresholds. Our experiments show good results for delta=0
     for all attributes. Note that w.shape[1]>delta.shape 
     """
     g,m = w.shape
     pscale = pd.DataFrame([[1 if w.iloc[i,j]> delta[j] else 0 for j in range(m)] for i in range(g)]
                           ,index =w.index
                           ,columns= ["+%i"%(int(c)) for c in w])
     nscale = pd.DataFrame([[1 if w.iloc[i,j]<= delta[j] else 0 for j in range(m)] for i in range(g)]
                           ,index =w.index
                           ,columns= ["-%i"%(int(c)) for c in w])
     scale = pd.concat([pscale,nscale],axis=1)
     return scale

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Model

f = open("imagenet-classes.txt", "r")
imagenet_classes = [c[:-1] for c in f]

def compute_class_views(model, name, lhl=-1,data="imagenet",classes=imagenet_classes):
    """Computes the many-valued and symbolic conceptual view for each model.
          - model is the neural network architecture and 
          - lhl is the index of the last hidden layer.
    """
    if model.layers[lhl].bias is not None:
        model_w,model_b = model.layers[lhl].trainable_weights
    else:
        model_w, = model.layers[lhl].trainable_weights
    # MobilNet needs to be reshaped
    if len(model_w.shape)==4:
        model_w = model_w.numpy().reshape(model_w.shape[2:4])
        model_b = model_b.numpy().reshape(model_b.shape[2:4])

    model_w = pd.DataFrame(model_w.numpy().T,index=classes)
    model_w_bin = make_scale(model_w)

    if model.layers[lhl].bias is not None:
        model_b = pd.DataFrame(model_b.numpy().T,index=classes)
        model_b.to_csv(f"scales/{data}_class/{name}_b.csv")    

    model_w.to_csv(f"scales/{data}_class/{name}_w.csv")
    model_w_bin.to_csv(f"scales/{data}_class/{name}_w_bin.csv")
    return model_w, model_w_bin

def embedding(model,lhl=-1):
    if lhl==-3: # for mobilnetv1
        out = tf.keras.layers.Reshape((1024,))(model.layers[lhl-1].output)
        E = Model(model.layers[0].input,out)
    else:
        E = Model(model.layers[0].input,model.layers[lhl-1].output)
    return E

def compute_object_views(model,name,data_folder,lhl=-1,data="imagenet"):
    E = embedding(model,lhl)

    input_shape = E.input_shape[1:3]

    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow_from_directory(data_folder, target_size=input_shape, shuffle=False, subset=None, classes=None)

    model_o = pd.DataFrame(E.predict(test_gen))
    model_o.to_csv(f"scales/{data}_obj/{name}_o.csv")

    model_o_bin = make_scale(model_o)
    model_o_bin.to_csv(f"scales/{data}_obj/{name}_o_bin.csv")
    return model_o, model_o_bin

def compute_model_predictions(model,name,data_folder,data="imagenet"):
    input_shape = model.input_shape[1:3]

    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow_from_directory(data_folder, target_size=input_shape, shuffle=False, subset=None, classes=None)

    model_pred = pd.DataFrame(model.predict(test_gen)).apply(lambda r : list(r).index(r.max()),axis=1)
    model_pred.to_csv(f"scales/{data}_class/{name}_pred.csv")
    return model_pred

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

base_32 = network(input_shape=input_shape, num_classes=num_classes)
base_32.load_weights(f"{output_dir}/base-32-tanh/model.h5")
base_16 = network16(input_shape=input_shape, num_classes=num_classes)
base_16.load_weights(f"{output_dir}/base-16-tanh/model.h5")

vgg16_32 = tf.keras.models.load_model("fruit_models/vgg16-32-tanh/model.h5")
vgg16_16 = tf.keras.models.load_model("fruit_models/vgg16-16-tanh/model.h5")

resnet50_32 = tf.keras.models.load_model("fruit_models/resnet-32-tanh/model.h5")
resnet50_16 = tf.keras.models.load_model("fruit_models/resnet-16-tanh/model.h5")

incv3_32 = tf.keras.models.load_model("fruit_models/incv3-32-tanh/model.h5")
incv3_16 = tf.keras.models.load_model("fruit_models/incv3-16-tanh/model.h5")

effb0_32 = tf.keras.models.load_model("fruit_models/effb0-32-tanh/model.h5")
effb0_16 = tf.keras.models.load_model("fruit_models/effb0-16-tanh/model.h5")

data_folder = "image-data/fruit360/"

compute_class_views(base_32,"base_32",data="fruit",classes=labels)
compute_object_views(base_32,"base_32",data_folder=data_folder,data="fruit")
compute_model_predictions(base_32,"base_32",data_folder=data_folder,data="fruit")

compute_class_views(base_16,"base_16",data="fruit",classes=labels)
compute_object_views(base_16,"base_16",data_folder=data_folder,data="fruit")
compute_model_predictions(base_16,"base_16",data_folder=data_folder,data="fruit")

compute_class_views(vgg16_32,"vgg16_32",data="fruit",classes=labels)
compute_object_views(vgg16_32,"vgg16_32",data_folder=data_folder,data="fruit")
compute__model_predictions(vgg16_32,"vgg16_32",data_folder=data_folder,data="fruit")

compute_class_views(vgg16_16,"vgg16_16",data="fruit")
compute_object_views(vgg16_16,"vgg16_16",data_folder=data_folder,data="fruit")
compute_model_predictions(vgg16_16,"vgg16_16",data_folder=data_folder,data="fruit")

compute_class_views(resnet50_32,"resnet50_32",data="fruit",classes=labels)
compute_object_views(resnet50_32,"resnet50_32",data_folder=data_folder,data="fruit")
compute_model_predictions(resnet50_32,"resnet50_32",data_folder=data_folder,data="fruit")

compute_class_views(resnet50_16,"resnet50_16",data="fruit")
compute_object_views(resnet50_16,"resnet50_16",data_folder=data_folder,data="fruit")
compute_model_predictions(resnet50_16,"resnet50_16",data_folder=data_folder,data="fruit")

compute_class_views(incv3_32,"incv3_32",data="fruit",classes=labels)
compute_object_views(incv3_32,"incv3_32",data_folder=data_folder,data="fruit")
compute_model_predictions(incv3_32,"incv3_32",data_folder=data_folder,data="fruit")

compute_class_views(incv3_16,"incv3_16",data="fruit")
compute_object_views(incv3_16,"incv3_16",data_folder=data_folder,data="fruit")
compute_model_predictions(incv3_16,"incv3_16",data_folder=data_folder,data="fruit")

compute_class_views(effb0_32,"effb0_32",data="fruit",classes=labels)
compute_object_views(effb0_32,"effb0_32",data_folder=data_folder,data="fruit")
compute_model_predictions(effb0_32,"effb0_32",data_folder=data_folder,data="fruit")

compute_class_views(effb0_16,"effb0_16",data="fruit")
compute_object_views(effb0_16,"effb0_16",data_folder=data_folder,data="fruit")
compute_model_predictions(effb0_16,"effb0_16",data_folder=data_folder,data="fruit")
