import os
import pandas as pd
import numpy as np

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

if not os.path.exists("scales/imagenet_class"):
    os.makedirs("scales/imagenet_class")
if not os.path.exists("scales/imagenet_obj"):
    os.makedirs("scales/imagenet_obj")

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

data_folder = "image-data/imagenet/"

from tensorflow.keras.applications import VGG16, VGG19
vgg16 = VGG16(include_top=True, weights="imagenet",classes=1000)
vgg19 = VGG19(include_top=True, weights="imagenet",classes=1000)

compute_class_views(vgg16,"vgg16")
compute_class_views(vgg19,"vgg19")

compute_object_views(vgg16,"vgg16",data_folder=data_folder)
compute_object_views(vgg19,"vgg19",data_folder=data_folder)

compute_model_predictions(vgg16,"vgg16",data_folder=data_folder)
compute_model_predictions(vgg19,"vgg19",data_folder=data_folder)

from tensorflow.keras.applications.inception_v3 import InceptionV3
incv3 = InceptionV3(include_top=True, weights="imagenet",classes=1000)
compute_class_views(incv3,"incv3")
compute_object_views(incv3,"incv3",data_folder=data_folder)
compute_model_predictions(incv3,"incv3",data_folder=data_folder)

from tensorflow.keras.applications.efficientnet import EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7
effb0 = EfficientNetB0(include_top=True, weights="imagenet",classes=1000)
effb1 = EfficientNetB1(include_top=True, weights="imagenet",classes=1000)
effb2 = EfficientNetB2(include_top=True, weights="imagenet",classes=1000)
effb3 = EfficientNetB3(include_top=True, weights="imagenet",classes=1000)
effb4 = EfficientNetB4(include_top=True, weights="imagenet",classes=1000)
effb5 = EfficientNetB5(include_top=True, weights="imagenet",classes=1000)
effb6 = EfficientNetB6(include_top=True, weights="imagenet",classes=1000)
effb7 = EfficientNetB7(include_top=True, weights="imagenet",classes=1000)

compute_class_views(effb0,"effb0")
compute_class_views(effb1,"effb1")
compute_class_views(effb2,"effb2")
compute_class_views(effb3,"effb3")
compute_class_views(effb4,"effb4")
compute_class_views(effb5,"effb5")
compute_class_views(effb6,"effb6")
compute_class_views(effb7,"effb7")

compute_object_views(effb0,"effb0",data_folder=data_folder)
compute_object_views(effb1,"effb1",data_folder=data_folder)
compute_object_views(effb2,"effb2",data_folder=data_folder)
compute_object_views(effb3,"effb3",data_folder=data_folder)
compute_object_views(effb4,"effb4",data_folder=data_folder)
compute_object_views(effb5,"effb5",data_folder=data_folder)
compute_object_views(effb6,"effb6",data_folder=data_folder)
compute_object_views(effb7,"effb7",data_folder=data_folder)

compute_model_predictions(effb0,"effb0",data_folder=data_folder)
compute_model_predictions(effb1,"effb1",data_folder=data_folder)
compute_model_predictions(effb2,"effb2",data_folder=data_folder)
compute_model_predictions(effb3,"effb3",data_folder=data_folder)
compute_model_predictions(effb4,"effb4",data_folder=data_folder)
compute_model_predictions(effb5,"effb5",data_folder=data_folder)
compute_model_predictions(effb6,"effb6",data_folder=data_folder)
compute_model_predictions(effb7,"effb7",data_folder=data_folder)

from tensorflow.keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
densenet121 = DenseNet121(include_top=True, weights="imagenet",classes=1000)
densenet169 = DenseNet169(include_top=True, weights="imagenet",classes=1000)
densenet201 = DenseNet201(include_top=True, weights="imagenet",classes=1000)

compute_class_views(densenet121,"densenet121")
compute_class_views(densenet169,"densenet169")
compute_class_views(densenet201,"densenet201")

compute_object_views(densenet121,"densenet121",data_folder=data_folder)
compute_object_views(densenet169,"densenet169",data_folder=data_folder)
compute_object_views(densenet201,"densenet201",data_folder=data_folder)

compute_model_predictions(densenet121,"densenet121",data_folder=data_folder)
compute_model_predictions(densenet169,"densenet169",data_folder=data_folder)
compute_model_predictions(densenet201,"densenet201",data_folder=data_folder)

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
incresnetv2 = InceptionResNetV2(include_top=True, weights="imagenet",classes=1000)

compute_class_views(incresnetv2,"incresnetv2")
compute_object_views(incresnetv2,"incresnetv2",data_folder=data_folder)
compute_model_predictions(incresnetv2,"incresnetv2",data_folder=data_folder)

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
mobilnetv1 = MobileNet(include_top=True, weights="imagenet",classes=1000)
mobilnetv2 = MobileNetV2(include_top=True, weights="imagenet",classes=1000)

compute_class_views(mobilnetv1,"mobilnetv1",lhl=-3)
compute_class_views(mobilnetv2,"mobilnetv2")

compute_object_views(mobilnetv1,"mobilnetv1",lhl=-3,data_folder=data_folder)
compute_object_views(mobilnetv2,"mobilnetv2",data_folder=data_folder)

compute_model_predictions(mobilnetv1,"mobilnetv1",data_folder=data_folder)
compute_model_predictions(mobilnetv2,"mobilnetv2",data_folder=data_folder)

from tensorflow.keras.applications.nasnet import NASNetMobile,NASNetLarge
nasnetlarge = NASNetLarge(include_top=True, weights="imagenet",classes=1000)
nasnetmobile = NASNetMobile(include_top=True, weights="imagenet",classes=1000)

compute_class_views(nasnetlarge,"nasnetlarge")
compute_class_views(nasnetmobile,"nasnetmobile")

compute_object_views(nasnetlarge,"nasnetlarge",data_folder=data_folder)
compute_object_views(nasnetmobile,"nasnetmobile",data_folder=data_folder)

compute_model_predictions(nasnetlarge,"nasnetlarge",data_folder=data_folder)
compute_model_predictions(nasnetmobile,"nasnetmobile",data_folder=data_folder)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet101V2,ResNet152V2,ResNet50V2

resnet50 = ResNet50(include_top=True, weights="imagenet",classes=1000)
resnet101v2 = ResNet101V2(include_top=True, weights="imagenet",classes=1000)
resnet152v2 = ResNet152V2(include_top=True, weights="imagenet",classes=1000)
resnet50v2 = ResNet50V2(include_top=True, weights="imagenet",classes=1000)

compute_class_views(resnet50,"resnet50")
compute_class_views(resnet101v2,"resnet101v2")
compute_class_views(resnet152v2,"resnet152v2")
compute_class_views(resnet50v2,"resnet50v2")

compute_object_views(resnet50,"resnet50",data_folder=data_folder)
compute_object_views(resnet101v2,"resnet101v2",data_folder=data_folder)
compute_object_views(resnet152v2,"resnet152v2",data_folder=data_folder)
compute_object_views(resnet50v2,"resnet50v2",data_folder=data_folder)

compute_model_predictions(resnet50,"resnet50",data_folder=data_folder)
compute_model_predictions(resnet101v2,"resnet101v2",data_folder=data_folder)
compute_model_predictions(resnet152v2,"resnet152v2",data_folder=data_folder)
compute_model_predictions(resnet50v2,"resnet50v2",data_folder=data_folder)

from tensorflow.keras.applications.xception import Xception
xception = Xception(include_top=True, weights="imagenet",classes=1000)

compute_class_views(xception,"xception")
compute_object_views(xception,"xception",data_folder=data_folder)
compute_model_predictions(xception,"xception",data_folder=data_folder)
