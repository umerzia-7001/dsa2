# -*- coding: utf-8 -*- 
"""

 ! activate py36 && python source/run_inference.py  run_predict  --n_sample 1000  --model_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data /data/input/train/
 

"""
import warnings
warnings.filterwarnings('ignore')
import sys
import gc
import os
import logging
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import json
import importlib

# from tqdm import tqdm_notebook
import cloudpickle as pickle


from diskcache import Cache
cache = Cache('d:/ztmp/diskcache_db.cache')
cache.reset('size_limit', int(2e9))



#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)
package_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
print(package_path)
sys.path.append( package_path)
import util_feature


####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m

    ### Implement Logging
    print(sjump, sspace, s, sspace, flush=True)


def save(path, name_list, glob):
    import pickle, os
    os.makedirs(path, exist_ok=True)
    for t in name_list:
        log(t)
        pickle.dump(glob[t], open(f'{t}', mode='wb'))


def load(name):
    import cloudpickle as pickle
    return pickle.load(open(f'{name}', mode='rb'))


####################################################################################################
####################################################################################################
def load_dataset(path_data, n_sample=-1, colid="jobId"):
    log('loading', colid, path_data)
    df = pd.read_csv(path_data + "/features.zip")
    df = df.set_index(colid)
    if n_sample > 0:
        df = df.iloc[:n_sample, :]        
    try:
        dfy = pd.read_csv(path_data + "/target_values.zip")
        df = df.join(dfy.set_index(colid), on=colid, how='left', )
    except:
        pass
    return df


# @cache.memoize(typed=True,  tag='fib')  ### allow caching results
def preprocess(df, path_pipeline="data/pipeline/pipe_01/"):
    """
      FUNCTIONNAL approach is used for pre-processing, so the code can be EASILY extensible to PYSPPARK.
      PYSPARK  supports better UDF, lambda function
    """

    log("########### Load preprocessor data  ##################################")
    colid  = load(f'{path_pipeline}/colid.pkl')
    coly   = load(f'{path_pipeline}/coly.pkl')
    colcat = load(f'{path_pipeline}/colcat.pkl')
    colcat_onehot  = load(f'{path_pipeline}/colcat_onehot.pkl')
    colcat_bin_map = load(f'{path_pipeline}/colcat_bin_map.pkl')

    colnum = load(f'{path_pipeline}/colnum.pkl')
    colnum_binmap = load(f'{path_pipeline}/colnum_binmap.pkl')
    colnum_onehot = load(f'{path_pipeline}/colnum_onehot.pkl')

    log("###### Colcat Pereprocess ############################################")
    df_cat_hot, _ = util_feature.pd_col_to_onehot(df[colcat],
                                                  colname=colcat,
                                                  colonehot=colcat_onehot, return_val="dataframe,param")

    log(df_cat_hot[colcat_onehot].head(5))

    log("###### Colcat As integer  ############################################")
    df_cat_bin, _ = util_feature.pd_colcat_toint(df[colcat],
                                                 colname=colcat,
                                                 colcat_map=colcat_bin_map, suffix="_int")
    colcat_bin = list(df_cat_bin.columns)

    log("###### Colnum Preprocess   ###########################################")
    df_num, _ = util_feature.pd_colnum_tocat(df, colname=colnum, colexclude=None,
                                             colbinmap=colnum_binmap,
                                             bins=-1, suffix="_bin", method="",
                                             return_val="dataframe,param")
    log(colnum_binmap)
    colnum_bin = [x + "_bin" for x in list(colnum_binmap.keys())]
    log(df_num[colnum_bin].head(5))

    ###### Map numerics bin to One Hot
    df_num_hot, _ = util_feature.pd_col_to_onehot(df_num[colnum_bin], colname=colnum_bin,
                                                  colonehot=colnum_onehot, return_val="dataframe,param")
    log(df_num_hot[colnum_onehot].head(5))

    log("####### colcross cross features   ######################################")
    df_onehot = df_cat_hot.join(df_num_hot, on=colid, how='left')

    colcat_onehot2 = [x for x in colcat_onehot if 'companyId' not in x]
    # log(colcat_onehot2)
    colcross_single = colnum_onehot + colcat_onehot2
    df_onehot = df_onehot[colcross_single]
    dfcross_hot, colcross_pair = util_feature.pd_feature_generate_cross(df_onehot, colcross_single,
                                                                        pct_threshold=0.02,
                                                                        m_combination=2)
    log(dfcross_hot.head(2).T)
    colcross_onehot = list(dfcross_hot.columns)
    del df_onehot
    gc.collect()

    log("##### Merge data type together  :   ###################### ")
    dfmerge = pd.concat((df[colnum], df_num, df_num_hot,
                         df[colcat], df_cat_bin, df_cat_hot,
                         dfcross_hot
                         ), axis=1)
    col_merge = list(dfmerge.columns)
    del df
    gc.collect()

    log("###### Export columns group   ##########################################################")
    cols_family = {}
    for coli in ['coly', 'colid', "colnum", "colnum_bin", "colnum_onehot", "colcat", "colcat_bin",
                 "colcat_onehot", "colcross_onehot", 'col_merge']:
        cols_family[coli] = locals()[coli]

    return dfmerge, cols_family

####################################################################################################
####################################################################################################
def map_model(model_name):
    try :
       ##  'models.model_bayesian_pyro'   'model_widedeep'
       mod    = f'models.{model_name}'
       modelx = importlib.import_module(mod) 
       
    except :
        ### Al SKLEARN API
        #['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
       mod    = 'models.model_sklearn'
       modelx = importlib.import_module(mod) 
    
    return modelx



def predict(model_name, path_model, dfX, cols_family):
    """
    if model_name in ['ElasticNet', 'ElasticNetCV', 'LGBMRegressor', 'LGBMModel', 'TweedieRegressor', 'Ridge']:
        from models import model_sklearn as modelx

    elif model_name == 'model_bayesian_pyro':
        from models import model_bayesian_pyro as modelx

    elif model_name == 'model_widedeep':
        from models import model_widedeep as modelx
    """
    modelx = map_model(model_name)    
    modelx.reset()
    log(modelx, path_model)    
    #log(os.getcwd())
    sys.path.append( root)    #### Needed due to import source error    
    
    modelx.model = load(path_model + "/model/model.pkl")
    # stats = load(path_model + "/model/info.pkl")
    colsX = load(path_model + "/model/colsX.pkl")
    # coly  = load( path_model + "/model/coly.pkl"   )
    log(modelx.model.model)

    ### Prediction
    ypred = modelx.predict(dfX[colsX])

    return ypred




####################################################################################################
############CLI Command ############################################################################
def run_predict(model_name, path_model, path_data, path_output, n_sample=-1):
    path_output = root + path_output
    path_data = root + path_data
    path_model = root + path_model
    path_pipeline = path_model + "/pipeline/"
    log(path_data, path_model, path_output)

    df = load_dataset(path_data, n_sample)

    dfX, cols_family = preprocess(df, path_pipeline)
    
    ypred = predict(model_name, path_model, dfX, cols_family)

    log("Saving prediction", ypred.shape, path_output)
    os.makedirs(path_output, exist_ok=True)
    df[cols_family["coly"] + "_pred"] = ypred
    df.to_csv(f"{path_output}/prediction.csv")
    log(df.head(8))

    #####  Export Specific
    df[cols_family["coly"]] = ypred
    df[[cols_family["coly"]]].to_csv(f"{path_output}/test_salaries.csv")


def run_check(path_data, path_data_ref, path_model, path_output, sample_ratio=0.5):
    """
     Calcualata Dataset Shift before prediction.
    """
    path_output = root + path_output
    path_data = root + path_data
    path_data_ref = root + path_data_ref
    path_pipeline = root + path_model + "/pipeline/"

    os.makedirs(path_output, exist_ok=True)

    df1 = load_dataset(path_data_ref)
    dfX1, cols_family1 = preprocess(df1, path_pipeline)

    df2 = load_dataset(path_data)
    dfX2, cols_family2 = preprocess(df2, path_pipeline)

    colsX = cols_family1["colnum_bin"] + cols_family1["colcat_bin"]
    dfX1 = dfX1[colsX]
    dfX2 = dfX2[colsX]

    nsample = int(min(len(dfX1), len(dfX2)) * sample_ratio)
    metrics_psi = util_feature.dataset_psi_get(dfX2, dfX1,
                                               colsX, nsample=nsample, buckets=7, axis=0)
    metrics_psi.to_csv(f"{path_output}/prediction_features_metrics.csv")
    log(metrics_psi)


if __name__ == "__main__":
    import fire

    fire.Fire()