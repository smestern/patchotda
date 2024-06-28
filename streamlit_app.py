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
from patchOTDA.domain_adapt import PatchClampOTDA
from functools import partial
import ot.da
from ot.backend import get_backend
import umap
from streamlit_tree_select import tree_select
import xgboost as xgb

#MODELS 
MODELS = {
        'EMDLaplace (EMD based distance transport) - Unsupervised': {'model': ot.da.EMDLaplaceTransport, 'params': {'reg_lap': (0., 10., 10.), 'reg_src':(0., 10., 0.9), 'norm': ['median', None,  'max'], 'verbose': False}, 'Description':''},
        'UnbalancedSinkhornTransport (Optimal Transport) - Unsupervised - unevenly Sampled': {'model': ot.da.UnbalancedSinkhornTransport, 'params': {'reg_e': (0., 2., 0.1), 'reg_m': (0., 2., 0.1), 'max_iter': (10, 10000, 1000), 'norm': [None, 'median', 'max'], 'verbose': False}, 'Description':''},
        'JDOT (Joint Distribution Optimal Transport) - Semisupervised - unevenly Sampled': {'model': skada.JDOTC, 'params': {'alpha': (0, 10, 0.1), 'n_iter_max': (10, 10000, 1000)}, 'Description':''}}

CLASS_MODELS = {'Random Forest': {'model': RandomForestClassifier, 'params': {'n_estimators': (1, 1000, 100, 3), 'max_depth': [50, None, 3, 5, 25, 100, 200], 'min_samples_split':(1, 100, 2, 3), 'min_samples_leaf':(1, 100, 2), 'min_impurity_decrease':(0.0, 100., 0.0, 3)}, 'Description':''},
                'XGBoost': {'model': xgb.XGBClassifier, 'params': {'n_estimators': (1, 1000, 100, 3), 'max_depth':(1, 100, 2, 4), 'learning_rate':(0.0, 1.0, 0.1, 4)}, 'Description':''},
                
                "Logistic Regression": {'model': LogisticRegression, 'params': {'C': (0., 1., 1.0, 4), 'penalty': ['l2', 'l1', 'elasticnet']}, 'Description':''},
                "SVM": {'model': SVC, 'params': {'C': (0., 1., 1.0, 4), 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}, 'Description':''}}

SCALERS = {'Standard Scaler': StandardScaler, 'MinMax Scaler': MinMaxScaler}

# Page title
st.set_page_config(page_title='PATCHOTDA: Map My Spikes Challenge 2024', page_icon='')
st.title('PATCHOTDA: Map My Spikes Challenge 2024')


with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.markdown('''This app aims to provide a user friendly interface to intergrate patch-clamp datasets from variety of conditions and species. He we employ domain adaptation methods and Hierarchical classification to predict the cell types of the user data. The app uses the PatchOTDA library to perform the domain adaptation and the HiClass library to perform the hierarchical classification.  
        This app was created by Smestern for the Map My Spikes Challenge 2024: https://alleninstitute.org/events/mapmyspikes/  
        ''')

  st.markdown('**How to use the app?**')
  st.markdown('''One can upload their own data or use the example data sets provided. The user can select the reference data and the labels to be used for the label propagation. 
            Its recommended to select the only the reference labels that are relevant to the user data.
            E.g. If the user data is from Sst, subselect the Sst labels.
            Next select the domain adaptation model and the classifier model. For domain Adaptation, the user can select from the following models:
                - EMDLaplace (EMD based distance transport) - Fully Unsupervised optimal transport regularizated by the laplacian. Recommended for evenly sampled data from the same distribution (E.g. All Cortex data)
                - UnbalancedSinkhornTransport (Optimal Transport) - Fully Unsupervised optimal transport regularizated by the mass. Recommended for unevenly sampled data from the same distribution (E.g. All Cortex data, majority Sst in the sample data)
                - JDOT (Joint Distribution Optimal Transport) - Semi-supervised optimal transport regularizated by the laplacian. Recommended for unevenly sampled data. *Can be finicky* (E.g. Cortex and Thalamus data)

            ''')

  st.markdown('**NOTE: Using your own data**')
  st.markdown('''As it stands, csv files uploaded must have columns that match names exactly with the example data. See: [here](https://www.dropbox.com/scl/fi/xfuil96rdvnq4rl4kxyf2/MapMySpikes_data_PUBLIC-final.xlsx?rlkey=uemaxpyne7x9bb6ewlgnljq34&e=1&st=2v6b26cr&dl=0])
               This is a limitation of the current version of the app.   
              In the future, I intend to allow the user to indicate feature matches between their data and the example data.   
              Users may be interested in computing their own features and using the app to predict cell types based on these features.  
              I recommened the Allen institute's IPFX library for this purpose: [IPFX](https://ipfx.readthedocs.io/en/latest/).
              
              ''')
  st.markdown("""The app is still in development and may have bugs. If you encounter any issues, please let me know.""")



# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=0)
        dataset_selected = 'User Data'
        USER_DATA[dataset_selected] = {}
        #split into ephys and meta
        #df_ephys, df_meta = select_by_col(df, MMS_DATA['joint_feats']), not_select_by_col(df, MMS_DATA['joint_feats'])
        USER_DATA[dataset_selected]['feat_matching'] = {}
        with st.expander('Feature Matching'):
            for feature in MMS_DATA['joint_feats']:
                if feature not in df.columns:  
                    USER_DATA[dataset_selected]['feat_matching'][feature] = st.selectbox(f"Select feature for {feature}", ['None', *df.columns])
                else:
                    USER_DATA[dataset_selected]['feat_matching'][feature] = st.selectbox(f"Select feature for {feature}", df.columns, index=int(np.where(df.columns == feature)[0][0]))
        
    # Select example data
    st.markdown('**1.2. Use example data**')
    dataset_selected = st.selectbox('Select data to map', EXAMPLE_DATA_)
    example_data = st.toggle('Load example data')


    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 50, 5)

    st.subheader('2.1. Reference Data')
    
    ref_data_name = st.selectbox('Reference data', REF_DATA_)
    ref_data = MMS_DATA[ref_data_name]
    st.text('Select the reference labels for the label propagation')
    with st.expander('See labels'):  
        if ref_data is not None:
            if ref_data_name == 'CTKE_M1':
                ref_labels = ['VISp_T']
            else:
                ref_labels = ['VISp_T', 'VISp_MET']
        labels= st.selectbox('Refence Labels', ref_labels)
        if labels == "VISp_T":
            return_tree = tree_select(VISp_T_nodes, "all")
        else:
            #make a tree select
            return_tree = tree_select(VISp_MET_nodes, "all")

    st.subheader('2.2. Model Selection')
    model = st.selectbox('Select Domain Adaptation model', list(MODELS.keys()))
    model_params = MODELS[model]['params']
    class_model = st.selectbox('Select Classifier model', list(CLASS_MODELS.keys()))
    class_model_params = CLASS_MODELS[class_model]['params'].copy()
    st.subheader('2.3. Learning Parameters')
    with st.expander('See parameters', expanded=False):
        #get from model
        st.write('Domain Adaptation model parameters:')
        if model is not None:
            for key, value in model_params.items():
                if isinstance(value, tuple):
                    model_params[key] = st.slider(key, value[0], value[1], value[2])
                elif isinstance(value, bool):
                    model_params[key] = st.checkbox(key, value)
                elif isinstance(value, list):
                    model_params[key] = st.selectbox(key, value)
        st.write('Classifier model parameters:') 
        grid_search = st.checkbox('Use Grid Search', False)
        if class_model is not None:
            for key, value in class_model_params.items():
                if isinstance(value, tuple):
                    class_model_params[key] = st.slider(key, value[0], value[1], value[2])
                elif isinstance(value, bool):
                    class_model_params[key] = st.checkbox(key, value)
                elif isinstance(value, list):
                    class_model_params[key] = st.selectbox(key, value)
                elif isinstance(value, int):
                    class_model_params[key] = st.slider(key, 0, 1000, value, 1)
                elif isinstance(value, float):
                    class_model_params[key] = st.slider(key, 0.0, 1.0, value, 0.01)
    

    st.subheader('2.4. General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_scale_seperate = st.checkbox('Scale data separately (not always rec. for domain adaptation)', False)
        if 'JDOT' in model:
            parameter_use_jdot = st.checkbox('Use JDOT classifier', False)

    st.header('3. Run Model')
    run_model = st.button('Run model')

# Initiate the model building process
if run_model: 
    with st.status("Running...", expanded=True) as status:
    
        st.write("Loading data...")
        #prepare the data, filter the reference data if needed
        if uploaded_file is not None:
            #df = pd.read_csv(uploaded_file, index_col=0)
            dataset_selected = 'User Data'
            #rename the columns based on the feature matching
            for key, value in USER_DATA[dataset_selected]['feat_matching'].items():
                if value != 'None':
                    df.rename(columns={value: key}, inplace=True)
            #split into ephys and meta
            df_ephys, df_meta = select_by_col(df, MMS_DATA['joint_feats']), not_select_by_col(df, MMS_DATA['joint_feats'])
            USER_DATA[dataset_selected] = {'ephys': df_ephys, 'meta': df_meta}
            st.write(df.head())
            st.write("User data loaded")
            #ask for matching features
            st.warning("please verify the features match the example data, rerun the app if they do not match.")
        elif example_data:
            df = MMS_DATA[dataset_selected]
            USER_DATA[dataset_selected] = MMS_DATA[dataset_selected]
        else:
            df = MMS_DATA[dataset_selected]
            USER_DATA[dataset_selected] = MMS_DATA[dataset_selected]
            st.warning('auto selecting example data')

        #prepare the ref
        ref_data = MMS_DATA[ref_data_name]
        ref_data_ephys, ref_data_meta = ref_data['ephys'], ref_data['meta']
        if return_tree is not None:
            if return_tree['checked'] == []:
                pass
            else:
                st.write("Filtering reference data ...")
                st.write(return_tree)
                idx = filter_MMS(ref_data_name, labels+"_3", return_tree)
                ref_data_ephys = ref_data_ephys.iloc[idx]
                ref_data_meta = ref_data_meta.iloc[idx]
                st.write("Reference data filtered")


        st.write("Preparing data...")

        MMS_DATA[ref_data_name]['pipeline'] = Pipeline([('imputer', KNNImputer()), ('scaler', StandardScaler())])
        USER_DATA[dataset_selected]['pipeline'] = Pipeline([('imputer', KNNImputer()), ('scaler', StandardScaler())])
        #USER_DATA[dataset_selected] = MMS_DATA[dataset_selected]
        Xt, Yt = ref_data_ephys, ref_data_meta[[labels+"_1_en", labels+"_2_en", labels+"_3_en"]]
        Yt = Yt.to_numpy()
        #remove outliers
        #get the user data
        Xs = USER_DATA[dataset_selected]['ephys']

        #get overlapping features
        Xt_features = Xt.columns.values
        Xs_features = Xs.columns.values
        common_features = np.intersect1d(Xt_features, Xs_features)
        Xs = Xs[common_features].to_numpy()
        Xt = Xt[common_features].to_numpy()
        st.write(f"Found {len(common_features)} common features between the reference and user data")
        Ys = np.full(Xs.shape[0], -1)
            
        
        Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(Xs, Ys, test_size=(100-parameter_split_size)/100, 
                                                                random_state=parameter_random_state)
        Xt_train, Xt_test, Yt_train, Yt_test = train_test_split(Xt, Yt, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)
        if parameter_scale_seperate:
            Xs_train = USER_DATA[dataset_selected]['pipeline'].fit_transform(Xs_train)
            Xs_test = USER_DATA[dataset_selected]['pipeline'].transform(Xs_test)
            Xt_train = MMS_DATA[ref_data_name]['pipeline'].fit_transform(Xt_train)
            Xt_test = MMS_DATA[ref_data_name]['pipeline'].transform(Xt_test)
        else:
            #fit first on vstacked data
            X = np.vstack([Xs_train, Xt_train])
            USER_DATA[dataset_selected]['pipeline'].fit(X)
            MMS_DATA[ref_data_name]['pipeline'].fit(X)
            Xs_train =  USER_DATA[dataset_selected]['pipeline'].transform(Xs_train)
            Xs_test =  USER_DATA[dataset_selected]['pipeline'].transform(Xs_test)
            Xt_train = MMS_DATA[ref_data_name]['pipeline'].transform(Xt_train)
            Xt_test = MMS_DATA[ref_data_name]['pipeline'].transform(Xt_test)

        st.write("Model training ... May take several minutes...")

        model = MODELS[model]['model'](**model_params)
        if 'JDOT' in model.__class__.__name__:
            model.fit(Xs_train, Xt_train, Ys_train, Yt_train[:,-1])
        elif 'EMDLaplace' in model.__class__.__name__:
            model.fit(Xs=Xs_train, Xt=Xt_train)
        elif "L1" in model.__class__.__name__:
            model.fit(Xs=Xs_train, Xt=Xt_train, Ys=Ys_train, Yt=Yt_train)
        elif 'UnbalancedSinkhornTransport' in model.__class__.__name__:
            model.fit(Xs=Xs_train, Xt=Xt_train)
            model.nx = get_backend(Xs_train)
        else:
            model.fit(Xs=Xs_train, Xt=Xt_train)
            #
            
        #now train our hiclass model
        st.write("Training nested model ...")
        Xs_translated = model.transform(Xs_train)
        Xs_test_translated = model.transform(Xs_test)
        hiclass = LocalClassifierPerLevel(CLASS_MODELS[class_model]['model'](**class_model_params))
        
        #grid search
        if grid_search:
            st.write("Grid searching ...")
            #JDOT is a bit of a nightmare so we will just use the nested model
            class dummy_cv_grid(LocalClassifierPerLevel): #this is a hack to get around the fact that the gridsearchcv does not like the LocalClassifierPerNode
                def __init__(self, **kwargs):
                    super().__init__(CLASS_MODELS[class_model]['model'](**kwargs))
                def set_params(self, **params):
                    return super().__init__(CLASS_MODELS[class_model]['model'](**params))

            #make a parameter grid

            grid_params = param_grid_from_dict(CLASS_MODELS[class_model]['params'])
            st.write(grid_params)
            searcher = GridSearchCV(CLASS_MODELS[class_model]['model'](), param_grid=grid_params, scoring=accuracy_score, cv=5, n_jobs=-1, verbose=1)
            searcher.fit(Xt_test, Yt_test[:,-1])
            hiclass = searcher.best_estimator_
            st.write(searcher.best_params_)
            hiclass.fit(Xt_train, Yt_train)
        else:
            hiclass.fit(Xt_train, Yt_train)


        st.write("Predicting ...")
        #if the model is JDOT we can predict the labels
        if 'JDOT' in model.__class__.__name__ and parameter_use_jdot:
            y_train_pred =  model.predict(Xt_train)
            y_test_pred =  model.predict(Xt_test)
            Ys_train_pred = model.predict(Xs_train)
            Ys_test_pred = model.predict(Xs_test)
            rf_results = pd.DataFrame({'Train': [accuracy_score(Ys_train, Ys_train_pred)],
                                    'Test': [accuracy_score(Ys_test, Ys_test_pred)]}, 
                                    index=['ACC', 'R2'])
        else: #make a nested rf
            #this is a bit of a nightmare but essentially for each sample
            #predict with the nested model
            y_train_pred = hiclass.predict(Xt_train)
            y_test_pred = hiclass.predict(Xt_test)
            Ys_train_pred = hiclass.predict(Xs_translated)
            Ys_test_pred = hiclass.predict(Xs_test_translated)
            
            #test the 3 levels of the model
            dict_results = {"Train":[], "Test":[]}
            for level in [0,1,2]:
                dict_results[f"Train"].append(accuracy_score(Yt_train[:,level].astype(np.int32), y_train_pred[:,level].astype(np.int32)))
                dict_results[f"Test"].append(accuracy_score(Yt_test[:,level].astype(np.int32), y_test_pred[:,level].astype(np.int32)))


                
            rf_results = pd.DataFrame(dict_results, 
                                    index=['ACC level 0', 'ACC level 1', 'ACC level 2'])
        
        st.write(rf_results)
        st.write("Embedding data ...")
    #embed the data using UMAP
    umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)

    #stack the data back together
    Xt_scaled_full = MMS_DATA[ref_data_name]['pipeline'].transform(Xt)
    Yt_pred_full = hiclass.predict(Xt_scaled_full)
    Yt_pred_full = np.array(Yt_pred_full)

    Xs_translated_full = model.transform(USER_DATA[dataset_selected]['pipeline'].transform(Xs))
    Ys_pred_full = hiclass.predict(Xs_translated_full)
    Ys_pred_full = np.array(Ys_pred_full)

    #label encode them back
    y_pred_full_str = np.full(Yt_pred_full.shape, 'unk')
    Ys_pred_full_str = np.full(Ys_pred_full.shape, 'unk')
    for level in [0,1,2]:
        le = MMS_DATA[ref_data_name][f'{labels}_LE_{level+1}']
        Yt_pred_full[:,level] = le.inverse_transform(Yt_pred_full[:,level].astype(np.int32))
        Ys_pred_full[:,level] = le.inverse_transform(Ys_pred_full[:,level].astype(np.int32))

    Xt_embedded = umap_model.fit_transform(Xt_scaled_full)
    Xs_embedded = umap_model.transform(Xs_translated_full)
    
    #plot the data
    st.write("Plotting data ...")
    plot_data = pd.DataFrame(np.vstack([Xs_embedded, Xt_embedded]), columns=['UMAP1', 'UMAP2'])
    plot_data['label'] = np.hstack([Ys_pred_full[:, 0], Yt_pred_full[:, 0]])
    plot_data['dataset'] = np.hstack([np.full(Xs_embedded.shape[0], 'User'), np.full(Xt_embedded.shape[0], 'Reference')])
    chart = (
        alt.Chart(plot_data)
        .mark_circle()
        .encode(x="UMAP1", y="UMAP2", shape="dataset", size='dataset', color="label", tooltip=["dataset", "label"])
        )

    st.altair_chart(chart, use_container_width=True)
    #prepare query data for the user
    st.write("Preparing data for export ...")
    query_data = pd.DataFrame(Xs, columns=common_features, index=USER_DATA[dataset_selected]['ephys'].index)
    query_data['label_level_1'] = Ys_pred_full[:,0]
    query_data['label_level_2'] = Ys_pred_full[:,1]
    query_data['label_level_3'] = Ys_pred_full[:,2]
        
    #save the data
    st.write("Saving data ...")
    #give the user the data
    st.download_button('Download Query Data', query_data.to_csv(), 'query_data.csv', 'text/csv')

    #make a joint df
    joint_df = pd.DataFrame(np.vstack([Xt, Xs]), columns=common_features, index=np.hstack([ref_data_ephys.index, USER_DATA[dataset_selected]['ephys'].index]))
    joint_df['label'] = np.hstack([Yt[:,0], Ys_pred_full[:,0]])
    st.download_button('Download Joint Data', joint_df.to_csv(), 'joint_data.csv', 'text/csv')

        
    status.update(label="Status", state="complete", expanded=False)



    
