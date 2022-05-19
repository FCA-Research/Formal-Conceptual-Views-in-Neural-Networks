import pandas as pd
import pysubgroup as ps

ontology = pd.read_csv("scales/fruit_ontology_scale.csv",index_col=0)
visual = pd.read_csv("scales/fruit_visual_scale.csv",index_col=0)

model = pd.read_csv("scales/fruit_class/base_16_w_bin.csv",index_col=0)
model_neg = model.loc[visual.index]
model = model.loc[visual.index,[c for c in model if "+" in c]]

def sg_detect(model,scale,target_name,a=1):
    searchdf = model.copy()
    searchdf[target_name] = scale[target_name]
    target = ps.BinaryTarget(target_name, True)
    searchspace = ps.create_selectors(searchdf, ignore=[target_name])
    task = ps.SubgroupDiscoveryTask(
        searchdf,
        target,
        searchspace,
        result_set_size=20,
        depth=10,
        qf=ps.StandardQF(a=a))
    result = ps.BeamSearch(beam_width=50).execute(task)
    return result.to_dataframe()

sg = sg_detect(model,visual,'F:Orange')
print(sg.iloc[[0,1,11]])

sg = sg_detect(model,ontology,'Äpfel')
print(sg.iloc[[0,1,4]])

sg = sg_detect(visual,model_neg,'-13',a=0.2)
print(sg.iloc[[1,17]])

sg = sg_detect(ontology,model_neg,'-13',a=0.2)
print(sg.iloc[[0]])
