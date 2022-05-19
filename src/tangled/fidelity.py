import pandas as pd
import os

obj_scales_dir = "scales/imagenet_obj/"
class_scales_dir = "scales/imagenet_class/"
models = [scale.split("_")[0] for scale in os.listdir(obj_scales_dir) 
                  if "csv" in scale and not "_bin" in scale]
to_obj_scale = lambda x: pd.read_csv(obj_scales_dir + x + "_o.csv",index_col=0)
to_class_scale = lambda x: pd.read_csv(class_scales_dir + x + "_w.csv",index_col=0)
to_bias_scale = lambda x: pd.read_csv(class_scales_dir + x + "_b.csv",index_col=0)
to_pred_scale = lambda x: pd.read_csv(class_scales_dir + x + "_pred.csv",index_col=0)

to_obj_bin_scale = lambda x: pd.read_csv(obj_scales_dir + x + "_o_bin.csv",index_col=0)
to_class_bin_scale = lambda x: pd.read_csv(class_scales_dir + x + "_w_bin.csv",index_col=0)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

    print("Train DTree")
    dt = DecisionTreeClassifier()
    dt.fit(model_w.values, model_w.index.values)

    dt_class_pred = dt.predict(model_o.loc[:,model_o.columns != "class"].values)
    dt_acc = accuracy_score(model_pred_idx,dt_class_pred)
    dt_f1 = f1_score(model_pred_idx,dt_class_pred,average="weighted")
    print(f"DTree Accuracy {dt_acc} KNN F1 score {dt_f1}")

    print("Train Eucliden")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(model_w.values, model_w.index.values)

    knn_class_pred = knn.predict(model_o.values)
    knn_acc = accuracy_score(model_pred_idx,knn_class_pred)
    knn_f1 = f1_score(model_pred_idx,knn_class_pred,average="weighted")
    print(f"Euclidean Accuracy {knn_acc} KNN F1 score {knn_f1}")

    print("Train Cos")
    cos_pred = cosine_pred(model_o.values,model_w)
    cos_acc = accuracy_score(model_pred_idx,cos_pred)
    cos_f1 = f1_score(model_pred_idx,cos_pred,average="weighted")
    print(f"COS Acc {cos_acc} f1 {cos_f1}")
    return knn_acc, cos_acc

for m in models:
    fidelity(to_obj_scale(m),to_class_scale(m),to_pred_scale(m))

for m in models:
    fidelity(to_obj_bin_scale(m),to_class_bin_scale(m),to_pred_scale(m))

import pandas as pd

obj_scales_dir = "scales/fruit_obj/"
class_scales_dir = "scales/fruit_obj/"
models = [scale.split("_")[0] for scale in os.listdir(obj_scales_dir) 
                  if "csv" in scale and not "_bin" in scale]
to_obj_scale = lambda x: pd.read_csv(obj_scales_dir + x + "_o.csv",index_col=0)
to_class_scale = lambda x: pd.read_csv(class_scales_dir + x + "_w.csv",index_col=0)
to_bias_scale = lambda x: pd.read_csv(class_scales_dir + x + "_b.csv",index_col=0)
to_pred_scale = lambda x: pd.read_csv(class_scales_dir + x + "_pred.csv",index_col=0)

to_obj_bin_scale = lambda x: pd.read_csv(obj_scales_dir + x + "_o_bin.csv",index_col=0)
to_class_bin_scale = lambda x: pd.read_csv(class_scales_dir + x + "_w_bin.csv",index_col=0)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

    print("Train DTree")
    dt = DecisionTreeClassifier()
    dt.fit(model_w.values, model_w.index.values)

    dt_class_pred = dt.predict(model_o.loc[:,model_o.columns != "class"].values)
    dt_acc = accuracy_score(model_pred_idx,dt_class_pred)
    dt_f1 = f1_score(model_pred_idx,dt_class_pred,average="weighted")
    print(f"DTree Accuracy {dt_acc} KNN F1 score {dt_f1}")

    print("Train Eucliden")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(model_w.values, model_w.index.values)

    knn_class_pred = knn.predict(model_o.values)
    knn_acc = accuracy_score(model_pred_idx,knn_class_pred)
    knn_f1 = f1_score(model_pred_idx,knn_class_pred,average="weighted")
    print(f"Euclidean Accuracy {knn_acc} KNN F1 score {knn_f1}")

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

for m in models:
    fidelity(to_obj_scale(m),to_class_scale(m),to_pred_scale(m))

for m in models:
    fidelity(to_obj_bin_scale(m),to_class_bin_scale(m),to_pred_scale(m))
    separation(to_class_bin_scale(m))
