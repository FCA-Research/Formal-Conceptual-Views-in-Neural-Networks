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

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import norm
from sklearn.metrics import accuracy_score,f1_score

def cosine_classify(o,model_w):
    cosine = [np.dot(o,model_w.loc[f])/(norm(o)*norm(model_w.loc[f])) for f in model_w.index]
    return model_w.index[cosine.index(max(cosine))]

def cosine_pred(model_o,model_w):
    # find index that has highest cosine sim
    pred = []
    clf = lambda x: cosine_classify(o,model_w)
    for o in model_o:
        cosine = [np.dot(o,model_w.loc[f])/(norm(o)*norm(model_w.loc[f])) for f in model_w.index]
        pred.append(model_w.index[cosine.index(max(cosine))])
    return pred

def fidelity(model_o,model_w,model_pred):
    model_pred_idx = model_pred["0"].apply(lambda i : model_w.index[i])
    print("Train 1NN")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(model_w.values, model_w.index.values)

    knn_class_pred = knn.predict(model_o.values)
    knn_acc = accuracy_score(model_pred_idx,knn_class_pred)
    knn_f1 = f1_score(model_pred_idx,knn_class_pred,average="weighted")
    print(f"KNN Accuracy {knn_acc} KNN F1 score {knn_f1}")

    print("Train Cos")
    cos_pred = cosine_pred(model_o.values,model_w)
    cos_acc = accuracy_score(model_pred_idx,cos_pred)
    cos_f1 = f1_score(model_pred_idx,cos_pred,average="weighted")
    print(f"COS Acc {cos_acc} f1 {cos_f1}")
    return knn_acc, cos_acc

def classes_seperation(model_w):
    unique = len({tuple(model_w.loc[i].values.tolist()) for i in model_w.index})
    separation = unique / model_w.shape[0]
    print(f"{unique} of {model_w.shape[0]} are distinct: separation: {separation} ")
    return separation

import pickle
import os

activation_list=["swish","tanh","relu","linear"]
neurons = [4,5,6,7,8,9]# 16,32,64,128,256,512

if not os.path.exists("scales/ablation"):
    os.makedirs("scales/ablation")

for a in activation_list:
    for n in neurons:
        # Setting
        print(a)
        print(n)

        # Computed Views
        weights = []
        embedd = []
        weights_bin = []
        embedd_bin = []
        pred = []

        # ten runs
        for i in range(10):
            print(f"run {i}")
            tf.random.set_seed(i)
            # Make model and train
            model = network(input_shape=input_shape, num_classes=num_classes,activation=a,neurons_second_last=(10+n)//2,neurons_last=n)
            train_and_evaluate_model(model, name=f"{a}-{n}-activation",epochs=20)

            # Compute views
            model_w, model_w_bin = compute_class_views(model,f"{a}-{n}-{i}-activation", lhl=-1,data="ablation")
            model_o, model_o_bin = compute_object_views(model,f"{a}-{n}-{i}-activation",test_gen,lhl=-1,data="ablation")
            model_pred = compute_model_predictions(model,f"{a}-{n}-{i}-activation",test_gen,data="ablation")

            # save results
            weights.append(model_w)
            weights_bin.append(model_w_bin)

            embedd.append(model_o)
            embedd_bin.append(model_o_bin)

            pred.append(model_pred)

        # Dump Models
        with open(f"scales/ablation_class/weights-{a}-{n}.pkl", 'wb') as fh:
            pickle.dump(weights, fh)
        with open(f"scales/ablation_class/weights-bin-{a}-{n}.pkl", 'wb') as fh:
            pickle.dump(weights_bin, fh)

        with open(f"scales/ablation_obj/O-{a}-{n}.pkl", 'wb') as fh:
            pickle.dump(embedd, fh)
        with open(f"scales/ablation_obj/O-bin-{a}-{n}.pkl", 'wb') as fh:
            pickle.dump(embedd_bin, fh)

        with open(f"scales/ablation_class/pred-{a}-{n}.pkl", 'wb') as fh:
            pickle.dump(pred, fh)

for a in activation_list:
    for n in neurons:
        weights = pickle.load(open(f"scales/ablation_class/weights-{a}-{n}.pkl",'rb'))
        weights_bin = pickle.load(open(f"scales/ablation_class/weights-bin-{a}-{n}.pkl",'rb'))

        embedd = pickle.load(open(f"scales/ablation_obj/O-{a}-{n}.pkl",'rb'))
        embedd_bin = pickle.load(open(f"scales/ablation_obj/O-bin-{a}-{n}.pkl",'rb'))      

        pred = pickle.load(open(f"scales/ablation_class/pred-{a}-{n}.pkl",'rb'))

        fid = [fidelity(embedd[i],weights[i],pred[i]) for i in range(len(weights))]
        fid_euclid = np.array([f[0] for f in fid])
        fid_cos = np.array([f[1] for f in fid])

        fid_bin = [fidelity(embedd_bin[i],weights_bin[i],pred[i]) for i in range(len(weights))]
        fid_bin_euclid = np.array([f[0] for f in fid_bin])
        fid_bin_cos = np.array([f[1] for f in fid_bin])

        sep = [separation(weights_bin[i])  for i in range(len(weights))]
        sep = np.array(sep)

        print(f"Activation {a} Neurons Many-Valued {n} Euclid Fid {fid_euclid.mean()}+-{fid.std()}")
        print(f"Activation {a} Neurons Symbolic {n} Euclid Fid {fid_bin_euclid.mean()}+-{fid_bin.std()}")

        print(f"Activation {a} Neurons Many-Valued {n} Cos Fid {fid_cos.mean()}+-{fid.std()}")
        print(f"Activation {a} Neurons Symbolic {n} Cos Fid {fid_bin_cos.mean()}+-{fid_bin.std()}")

        print(f"Activation {a} Neurons Symbolic {n} Separation {separation.mean()}+-{separation.std()}")
