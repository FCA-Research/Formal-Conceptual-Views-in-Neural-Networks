import pandas as pd
import os

import pandas as pd

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

def classes_seperation(model_w):
    unique = len({tuple(model_w.loc[i].values.tolist()) for i in model_w.index})
    separation = unique / model_w.shape[0]
    print(f"{unique} of {model_w.shape[0]} are distinct: separation: {separation} ")
    return separation

stats = pd.DataFrame(models)

stats["neurons"] = [to_class_scale(m).shape[1] for m in models]
stats.set_index(0,inplace=True)

stats["sep"] = [classes_seperation(to_class_scale(m)) for m in models]

stats["w"] = [(to_class_scale(m).values.mean(),to_class_scale(m).values.std()) 
              for m in models]
stats["b"] = [(to_bias_scale(m).values.mean(),to_class_scale(m).values.std()) 
              for m in models]
stats["o"] = [(to_object_scale(m).values.mean(),to_object_scale(m).values.std()) 
              for m in models]

stats.to_csv("scales/stats_imagenet.csv")
