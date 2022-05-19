from sklearn.metrics import accuracy_score
import os
import pandas as pd
import scipy as sp
import ot
import random

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

pairwise_fidelity = [[accuracy_score(to_pred_scale(m1).values
                                     ,to_pred_scale(m2).values) 
                      for m2 in models] for m1 in models]
fid_sim = pd.DataFrame(pairwise_fidelity,columns=models,index=models)
fid_sim.to_csv("scales/imagenet_fid_sim.csv")

sample_size = int(100000 * 0.1)

random.seed(42)
s = set(random.sample(range(100000),sample_size))
sample = pd.Series([i in s for i in range(100000)])

def gw_dist(m1,m2,C1=None,C2=None):
    """Compute the Gromov-Wasserstein distance between the two views. C
    are the euclidean distance matrices."""
    if C1 is None:
        M1 = to_obj_scale(m1)[sample]
        C1 = sp.spatial.distance.cdist(M1.values, M1.values,metric="euclidean")
        C1 /= C1.max()    
    if C2 is None:
        M2 = to_obj_scale(m2)[sample]
        C2 = sp.spatial.distance.cdist(M2.values, M2.values,metric="euclidean")
        C2 /= C2.max()    
    p = ot.unif(sample_size) # uniform
    q = ot.unif(sample_size) # uniform
    gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q
                                             ,'square_loss',  verbose=True, log=True)
    return log0["gw_dist"]

import multiprocessing as mp
num_cpu = 9

def gw_dist_iter(i):
    M1 = to_obj_scale(models[i])[sample]
    C1 = sp.spatial.distance.cdist(M1.values, M1.values,metric="euclidean")
    C1 /= C1.max()    
    return [0 if i == j else gw_dist(models[i],models[j],C1=C1) for j in range(len(models))]

pool = mp.Pool(num_cpu)
similarities = pool.map(gw_dist_iter, [i for i in range(len(models))])
pool.close()

S = pd.DataFrame(similarities,columns=models,index=models)
S.to_csv("imagenet_gromov.csv")
