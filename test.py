import streamlit as st
import pandas as pd
import numpy as np
#sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
#import XGBoost as xgb
#other
import altair as alt
import time
import zipfile
from utils import MMS_DATA, USER_DATA, EXAMPLE_DATA_, REF_DATA_, VISp_MET_nodes, VISp_T_nodes, filter_MMS, find_outlier_idxs, param_grid_from_dict, select_by_col, not_select_by_col
import os
from patchOTDA.external import skada
from patchOTDA.domainAdapt import PatchClampOTDA
from functools import partial
import ot.da
from ot.backend import get_backend
import umap
from streamlit_tree_select import tree_select
import xgboost as xgb


def test_filter_MMS():
    filter_MMS('VISp_Viewer', 'VISp_MET_3', {
  "checked": [
    "inhibitory",
    "Lamp5",
    "Lamp5-MET-1",
    "Lamp5-MET-2",
    "Pvalb",
    "Pvalb-MET-1",
    "Pvalb-MET-2",
    "Pvalb-MET-3",
    "Pvalb-MET-4",
    "Pvalb-MET-5",
    "Sncg",
    "Sncg-MET-1",
    "Sncg-MET-2",
    "Sncg-MET-3",
    "Sst",
    "Sst-MET-1",
    "Sst-MET-10",
    "Sst-MET-11",
    "Sst-MET-12",
    "Sst-MET-13",
    "Sst-MET-2",
    "Sst-MET-3",
    "Sst-MET-4",
    "Sst-MET-5",
    "Sst-MET-6",
    "Sst-MET-7",
    "Sst-MET-8",
    "Sst-MET-9",
    "Vip",
    "Vip-MET-1",
    "Vip-MET-2",
    "Vip-MET-3",
    "Vip-MET-4",
    "Vip-MET-5"
  ],
  "expanded": []
} )

def test_ub_sink():
    import ot
    from ot.datasets import make_2D_samples_gauss
    OT = ot.da.UnbalancedSinkhornTransport()
    Xs = make_2D_samples_gauss(n=1000, m=1000, sigma=[[2, 1], [1, 2]], random_state=42)
    Xt = make_2D_samples_gauss(n=1000, m=1000, sigma=[[2, 1], [1, 2]], random_state=42)[0]
    Xs = Xs.astype('float32')
    Xt = Xs + 0.5
    Xt = Xt.astype('float32')
    OT.fit(Xs, Xt)
    OT.transform(Xs)
    return OT

def test_JDOTC():
    from patchOTDA.external import skada
    from ot.datasets import make_data_classif
    from sklearn.model_selection import train_test_split
    import pickle as pkl
    OT = skada.JDOTC()
    Xs_train = pkl.load(open('Xs_train.pkl', 'rb'))
    Xt_train = pkl.load(open('Xt_train.pkl', 'rb'))
    Ys_train = pkl.load(open('Ys_train.pkl', 'rb'))
    Yt_train = pkl.load(open('Yt_train.pkl', 'rb'))
    OT = pkl.load(open('model.pkl', 'rb'))
    OT.fit(Xt_train, Xs_train,  Yt_train[:,-1], Ys_train)
    OT.transform(Xs_train)
    OT.transform(Xt_train)
    return OT


if __name__ == '__main__':
    test_JDOTC()
    #test_ub_sink()
    #test_filter_MMS()
    print("Passed all tests!")