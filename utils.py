import numpy as np
import pandas as pd
import pickle as pkl
import os
from patchOTDA.datasets import MMS_DATA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


EXAMPLE_DATA_ = ['Query1', 'Query2', 'Query3', 'CTKE_M1', 'VISp_Viewer']

REF_DATA_ = ['CTKE_M1', 'VISp_Viewer']
#build nodes for label propagation
VISp_MET_nodes = [#{'label':'excitatory', 'value': 'excitatory', 'color': '#FF0000', 'children':[]},
                   {'label':'inhibitory', 'value': 'inhibitory', 'color': '#0000FF', 'children':[]}]
VISp_T_nodes = [{'label':'excitatory', 'value': 'excitatory', 'color': '#FF0000', 'children':[]}, 
                {'label':'inhibitory', 'value': 'inhibitory', 'color': '#0000FF', 'children':[]}]

USER_DATA = {}

#parse the MET_nodes_out, hard coded for now
for data in REF_DATA_:
    mms_temp = MMS_DATA[data]['meta']
    #we want the column
    if 'VISp Viewer MET type' not in mms_temp.columns:
        continue
    labels = mms_temp['VISp Viewer MET type'].fillna('unk').values
    #get the unique labels
    unique_labels = np.unique(labels)
    for label in unique_labels:
        #split the label by -
        label_split = label.split('-')
        if len(label_split) == 1:
            #add to the root
            continue
        else:
            #add to the children, if not already there
            parent = label_split[0]
            child = '-'.join(label_split)
            
            #check if parent is in the nodes
            if parent not in [x['label'] for x in VISp_MET_nodes[0]['children']]:
                VISp_MET_nodes[0]['children'].append({'label':parent, 'value': parent, 'color': '#00FF00', 'children':[]})
            #check if child is in the nodes
            idx = [x['label'] for x in VISp_MET_nodes[0]['children']].index(parent)
            if child not in [x['label'] for x in VISp_MET_nodes[0]['children'][idx]['children']]:
                VISp_MET_nodes[0]['children'][idx]['children'].append({'label':child, 'value': child, 'color': '#00FF00'})
    

#parse the T_nodes_out, hard coded for now
for data in REF_DATA_:
    mms_temp = MMS_DATA[data]['meta']
    #we want the column
    if 'VISp Viewer T type' not in mms_temp.columns:
        continue
    labels = mms_temp['VISp Viewer T type'].fillna('unk').values
    #get the unique labels
    unique_labels = np.unique(labels)
    for label in unique_labels:
        #split the label by whitespace
        label_split = label.split(' ')
        if len(label_split) == 1:
            #add to the root
            continue
        else:
            if label_split[0] in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L6b', 'L2/3']:
                parent = label_split[0] + '_' + label_split[1]
                idx0 = 0
                child1 = '_'.join(label_split[0:3])
                child2 = '_'.join(label_split[0:4])
            else:
                parent = label_split[0]
                idx0 = 1
                child1 = '_'.join(label_split[0:2])
                child2 = '_'.join(label_split[0:3])
            
            #check if parent is in the nodes
            if parent not in [x['label'] for x in VISp_T_nodes[idx0]['children']]:
                VISp_T_nodes[idx0]['children'].append({'label':parent, 'value': parent, 'color': '#00FF00', 'children':[]})

            if child1 == parent:
                continue
            #check if child is in the nodes
            idx1 = [x['label'] for x in VISp_T_nodes[idx0]['children']].index(parent)
            if child1 not in [x['label'] for x in VISp_T_nodes[idx0]['children'][idx1]['children']]:
                VISp_T_nodes[idx0]['children'][idx1]['children'].append({'label':child1, 'value': child1, 'color': '#00FF00', 'children':[]})

            if child2 == child1:
                continue
            #check if child2 is a child of child1
            idx2 = [x['label'] for x in VISp_T_nodes[idx0]['children'][idx1]['children']].index(child1)
            if child2 not in [x['label'] for x in VISp_T_nodes[idx0]['children'][idx1]['children'][idx2]['children']]:
                VISp_T_nodes[idx0]['children'][idx1]['children'][idx2]['children'].append({'label':child2, 'value': child2, 'color': '#00FF00'})
            

#build a label encoder for each ref_data
for data in REF_DATA_:
    mms_temp = MMS_DATA[data]['meta']
    #we want the column
    if 'VISp Viewer MET type' in mms_temp.columns:
        for x in [1,2,3]:
            labels = mms_temp[f'VISp_MET_{x}'].fillna('unk').values
            #get the unique labels
            unique_labels = np.unique(labels)
            le = LabelEncoder()
            le.fit(unique_labels)
            MMS_DATA[data][f'VISp_MET_LE_{x}'] = le
            MMS_DATA[data]['meta'][f'VISp_MET_{x}_en'] = le.transform(labels)
    if 'VISp Viewer T type' in mms_temp.columns:
        for x in [1,2,3]:
            labels = mms_temp[f'VISp_T_{x}'].fillna('unk').values
            #get the unique labels
            unique_labels = np.unique(labels)
            le = LabelEncoder()
            le.fit(unique_labels)
            MMS_DATA[data][f'VISp_T_LE_{x}'] = le
            MMS_DATA[data]['meta'][f'VISp_T_{x}_en'] = le.transform(labels)

        
        




def filter_MMS(data, label_query, query):
    #get the data
    mms_temp = MMS_DATA[data]['meta']
    selected_labels = query['checked']
    #we want the column
    if label_query not in mms_temp.columns:
        return None
    labels = mms_temp[label_query].fillna('unk').values
    #get the indices where the label is in the query
    idx = np.nonzero([x in selected_labels for x in labels])[0]
    #get the indices
    return idx

def find_outlier_idxs(X, n_outliers=10):
    #find the outliers
    scale_and_imputer = Pipeline([('impute', SimpleImputer()), ('scaler',  StandardScaler())])
    x = scale_and_imputer.fit_transform(X)
    clf = IsolationForest(contamination=0.1)
    clf.fit(x)
    return np.where(clf.predict(x) == -1)[0] 