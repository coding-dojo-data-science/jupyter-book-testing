#!/usr/bin/env python
# coding: utf-8

# # Explaining Machine Learning Models

# - Office Hours for 022221FT
# - 05/19/21

# ## Overview

# - Fit an SVC and RandomForest model on our new data set. 
# - Discuss 3 different methods for interpreting the models' results and what features it used to make its predictions. 
#     - Permutation Importance
#     - Using `SHAP` and Shapely Values
#     - Using `LIME`

# ## Questions

# - [Google Doc](https://docs.google.com/document/d/1TFMucUZQPhGX6eGvhUpSlrByKHswgC30SUfybcuu6Hw/edit#)

# ## Notebook Headers for Study Group

# - ⭐️**EXPLAINING MACHINE LEARNING MODELS**⭐️
#     - All of today's content.
# -  📚**Shap Resources**
#     - Collection of Videos, Book Excerpts, and Blogs.

# ___

# # Predicing Drug Use

# ## Goal

# - To predict if someone is a heroin user and to use the model to gain insights into risk factors for using heroin. 

# ## Data

# - Drug Consumption Survey: 
#     - http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
# 
# - Data Set contains information about previous drug use for many classes of drugs, demographic data such as education, age, country, and personality scores for several metrics.
#     - Nscore
#     - Escore	
#     - Oscore	
#     - Ascore	
#     - Cscore	
#     - Impulsiveness	
#     - SS

# >- This dataset has already been heavily pre-processed to restore the orignal values of the dataset before the dataset providers encoded features. 
#     - See `reference`>`Feature Selection - data-renaming.ipynb`" [GitHub notebook Link](https://github.com/jirvingphd/dsc-phase-3-project/blob/9258a878234c98d13a48204131fa09eb9171f445/reference/Feature%20Selection%20-%20data-renaming.ipynb)

# ### Imports and Functions

# In[1]:


## Import pd, sns, plt, np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


## Preprocessing tools
from sklearn.model_selection import train_test_split,cross_val_predict,cross_validate
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE,SMOTENC


## Models & Utils
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from time import time


# In[3]:


# ## Changing Pandas Options to see full columns in previews and info
n=200
pd.set_option('display.max_columns',n)
pd.set_option("display.max_info_rows", n)
pd.set_option('display.max_info_columns',n)
# pd.set_option('display.float_format',lambda x: f"{x:.2f}")


# ### Modeling Functionx (WIP)_

# In[4]:


get_ipython().system('pip install tzlocal')


# In[5]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import project_functions as pf


# ## Preprocessing

# In[6]:


# df = pd.read_csv('https://raw.githubusercontent.com/jirvingphd/dsc-phase-3-project/main/data/drug_use_renamed_converted.csv')#'data/drug_use_renamed_converted.csv')
# df


# In[7]:


fname = 'Data/drug_use_renamed_converted.csv.gz'
# df.to_csv(fname,compression='gzip',index=False)
df= pd.read_csv(fname)
df


# In[8]:


df.info()


# In[9]:


columns= {0:'ID',1:'Age',2:'Gender',3:'Education',4:'Country',5:'Ethnicity',
         6:'Nscore',7:'Escore',8:'Oscore',9:"Ascore",10:'Cscore',
         11:'Impulsiveness',12:'SS'}

drug_dict = {13:'Alcohol', 
              14: 'Amphet', 
              15: 'Amyl', 
              16: 'Benzos', 
              17: 'Caff', 
              18: 'Cannabis', 
              19: 'Choc', 
              20: 'Coke',
              21: 'Crack', 
              22: 'Ecstasy',
              23: 'Heroin', 
              24: 'Ketamine', 
              25: 'LegalH', 
              26: 'LSD',
              27: 'Meth',
              28: 'Mushrooms', 
              29: 'Nicotine', 
              30: "FakeDrugSemeron",
              31: 'VSA'}

all_columns = {**columns, **drug_dict}

drug_cols = list(drug_dict.values())
drug_cols


# ### Identifying Columns for Preprocessing

# In[10]:


object_cols = list(df.drop(columns=drug_cols).select_dtypes('object').columns)
object_cols


# In[11]:


## Column Lists
ordinal_cols = ['Age','Education']
onehot_cols = ['Gender','Country', 'Ethnicity']


# In[12]:


df.select_dtypes('object').drop(columns=[*drug_cols,*ordinal_cols,*onehot_cols])


# ## Feature Engineering

# ### How to treat drug cols?

# - Either encode as ordinal or bin into 3 bins ('never used','previous user','current user')

# In[13]:


df['Heroin'].value_counts()


# In[14]:


df['Heroin'].unique()


# ## Making New df for processing

# In[15]:


df2 = df.copy()


# ### Binning Drug Use

# In[16]:


druguse_cat_map = {'Never Used':'Non-User' , 
 'Used over a Decade Ago': 'Non-User',
 'Used in Last Decade':"User" ,
 'Used in Last Year': "User",
 'Used in Last Week': "User", 
 'Used in Last Day': "User",
 'Used in Last Month':"User"}
druguse_cat_map


# In[17]:


for col in drug_cols:
    df2[col] = df[col].replace(druguse_cat_map)
#     display(df2[col].value_counts(normalize=True, dropna=False))


# In[18]:


df2['Heroin'].value_counts(1)


# ### Encoding Categorical Features (Ordinal)

# In[19]:


ordinal_cols


# ### Age

# In[20]:


## Making age map
age_map = {'18-24': 20,
           '25-34':30,
           '35-44':40, 
           '45-54':50,
           '55-64':60,
           '65+':70}


# In[21]:


df2['Age'] = df['Age'].replace(age_map)#.value_counts(dropna=False)
df2['Age'].value_counts(dropna=False)


# ### Education

# In[22]:


df['Education'].value_counts(dropna=False)


# In[23]:


education_map = {"Left school before 16 years":0, 
                 "Left school at 16 years":1, 
                 "Left school at 17 years":2,
                 "Left school at 18 years":3,
                 "Some college or university, no certificate or degree":4,
                 "Professional certificate/ diploma":5,
                 "University degree":6, "Masters degree":7, "Doctorate degree":8}

df2["Education"] = df['Education'].replace(education_map)
df2['Education'].value_counts(dropna=False)


# In[ ]:





# In[ ]:





# In[24]:


df2


# # Preprocessing

# In[25]:


## Specifying root names of types of features to loop through and filter out from df
target_col = 'Heroin'
drop_cols = ['ID']
target_map = {'Non-User':0, 'User':1}

y = df2[target_col].map(target_map).copy()
X = df2.drop(columns=[target_col,*drop_cols]).copy()
y.value_counts(1,dropna=False)


# In[26]:


X_train,X_test,y_train,y_test = train_test_split(X,y)
X_train


# In[27]:


from sklearn import set_config
set_config(display='diagram')


# In[28]:


## saving list of numeric vs categorical feature
num_cols = list(X_train.select_dtypes('number').columns)
cat_cols = list(X_train.select_dtypes('object').columns)

## create pipelines and column transformer
num_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scale',MinMaxScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='constant',fill_value='MISSING')),
    ('encoder',OneHotEncoder(sparse=False,drop='first'))
])

print('# of num_cols:',len(num_cols))
print('# of cat_cols:',len(cat_cols))


# In[29]:


## COMBINE BOTH PIPELINES INTO ONE WITH COLUMN TRANSFORMER
preprocessor=ColumnTransformer(transformers=[
    ('num',num_transformer,num_cols),
    ('cat',cat_transformer,cat_cols)])

preprocessor


# In[30]:


## Fit preprocessing pipeline on training data and pull out the feature names and X_cols
preprocessor.fit(X_train)

## Use the encoder's .get_feature_names
cat_features = list(preprocessor.named_transformers_['cat'].named_steps['encoder']\
                            .get_feature_names(cat_cols))
X_cols = num_cols+cat_features


# In[31]:


## Transform X_traian,X_test and remake dfs
X_train_df = pd.DataFrame(preprocessor.transform(X_train),
                          index=X_train.index, columns=X_cols)
X_test_df = pd.DataFrame(preprocessor.transform(X_test),
                          index=X_test.index, columns=X_cols)

## Tranform X_train and X_test and make into DataFrames
X_train_df


# In[32]:


y.value_counts(1)


# ## Resampling with SMOTENC

# In[33]:


y_train.value_counts(1)


# In[34]:


## Save list of trues and falses for each cols
smote_feats = [False]*len(num_cols) +[True]*len(cat_features)
# smote_feats


# In[35]:


## resample training data
smote = SMOTENC(smote_feats)
X_train_sm,y_train_sm = smote.fit_resample(X_train_df,y_train)
y_train_sm.value_counts()


# # MODELING

# #### Setting `train_test_list`

# In[36]:


### SAVING XY DATA TO LIST TO UNPACK
train_test_list = [X_train_sm,y_train_sm,X_test_df,y_test]


# ## Linear SVC

# In[37]:


svc_linear = SVC(kernel='linear',C=1)
pf.fit_and_time_model(svc_linear,*train_test_list)


# ## RandomForest

# In[38]:


rf = RandomForestClassifier()
pf.fit_and_time_model(rf,*train_test_list)


# In[39]:


pf.get_importance(rf,X_test_df);


# # ⭐️**EXPLAINING MACHINE LEARNING MODELS**⭐️

# ## Overview

# - We will discuss/demo 3 methods of interpreting machine learning models. 
#     1. Using Permutation Importance (from scikit-learn)
#     2. Using `SHAP` model explainers
#     3. Using `LIME` instance explainers

# ## Permutation Importance

# > Permutation Importances will iteratively shuffle the rows of a single feature at a time to asses the model's change in performance with that feature's relationship with the target disrupted. 
# - https://scikit-learn.org/stable/modules/permutation_importance.html
# 
# ```python
# from sklearn.inspection import permutation_importance
# r = permutation_importance(svc_linear, X_test_df, y_test, n_repeats=30)
# r.keys()
# ```
# - Interesting Observation: 
#     - permutation_importance takes a `scoring` argument!
# 
# > "**Warning Features that are deemed of low importance for a bad model (low cross-validation score) could be very important for a good model.** Therefore it is always important to evaluate the predictive power of a model using a held-out set (or better with cross-validation) prior to computing importances. Permutation importance does not reflect to the intrinsic predictive value of a feature by itself but how important this feature is for a particular model."

# In[40]:


from sklearn.inspection import permutation_importance


# In[41]:


## Permutation importance takes a fit mode and test data. 
r = permutation_importance(svc_linear, X_test_df, y_test,
                           n_repeats=30,scoring='f1')
r.keys()


# In[42]:


r['importances_mean']


# In[43]:


## can make the mean importances into a series
svc_importances = pd.Series(r['importances_mean'],index=X_train_df.columns,
                           name = 'svc permutation importance')
svc_importances


# In[44]:


r = permutation_importance(rf, X_test_df, y_test, n_repeats=30, scoring='f1')
rf_importances = pd.Series(r['importances_mean'],index=X_test_df.columns,
                          name= 'rf permutation importance')
rf_importances


# In[45]:


embedded_importances = pf.get_importance(rf,X_test_df)#,plot=False)
embedded_importances.name ='rf.feature_importances_'
embedded_importances


# In[46]:


def compare_importances(*importances,sort_index=True,sort_col=0,show_bar=False):
    """Accepts Series of feature importances to concat.
    
    Args:
        *importances (Seires): seires to concat (recommended to pre-set names of Series)
        sort_index (bool, default=True): return series sorted by index. 
                            If False, sort seires by sort_col  #
        sort_col (int, default=0): If sort_index=False, sort df by this column #
        show_bar (bool, default=False): If show_bar, returns a pandas styler instead of df
                                        with the importances plotted as bar graphs
        
    Returns:
        DataFrame: featutre importances     
    
        """
    ## Concat Importances
    compare_importances = pd.concat(importances,axis=1)
    
    ## Sort DF by index or by sort_col
    if sort_index:
        compare_importances = compare_importances.sort_index()
    else:
        sort_col_name = compare_importances.columns[sort_col]
        compare_importances= compare_importances.sort_values(sort_col_name,ascending=False)
        
    ## If show bar, return pandas styler with in-cell bargraphs
    if show_bar:
        return compare_importances.style.bar().set_caption('Feature Importances')
    else:
        return compare_importances


# In[47]:


## Compare embedded feature importance vs permutation importance
comp = compare_importances(embedded_importances,rf_importances,#svc_importances,
                          show_bar=True,sort_index=False,sort_col=0)
comp


# In[48]:


# df['']


# In[49]:


## Compare embedded feature importance vs permutation importance
comp = compare_importances(embedded_importances,rf_importances,svc_importances,
                          show_bar=True,sort_col=-1,sort_index=False)
comp


# ## Using SHAP and Shapely Values for Model Interpretation

# ###  📚**Shap Resources**

# >- SHAP (SHapley Additive exPlanations)) 
#     - [Repository](https://github.com/slundberg/shap)
#     - [Documentation](https://shap.readthedocs.io/en/latest/?badge=latest)
#         - Install via pip or conda.
#   
# 
# - SHAP uses game theory to calcualte Shapely values for each feature in the dataset. 
# - Shapely values are calculated by iteratively testing each feature's contribution to the model by comparing the model's  performance with vs. without the feature. (The "marginal contribution" of the feature to the model's performance).

# 
# 
# #### Papers, Book Excerpts, and  Blogs
# - [White Paper on Shapely Values](https://arxiv.org/abs/1705.07874)
#     
# - [Intepretable Machine Learning Book - Section on SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)
#     
# - Towards Data Science Blog Posts:
#     - [Explain Your Model with SHAP Values](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d)
# 
#     - [Explain Any Model with SHAP KernelExplaibner](https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a)
# 
# 
# 
# 
# 
# #### Videos/Talks:
# - Explaining Machine Learning Models (in general).
#     - ["Open the Black Box: an intro to Model Interpretability with LIME and SHAP](https://youtu.be/C80SQe16Rao)
# - Understanding Shapely/SHAP Values:
#     - [AI Simplified: SHAP Values in Machine Learning ](https://youtu.be/Tg8aPwPPJ9c)- (Intuitive Explanation)
#     - [Explainable AI explained! | #4 SHAP  ](https://youtu.be/9haIOplEIGM)- (Math Calculation Explanation)
# 

# ### How to Use SHAP

# - Uses game theory to explain feature importance and how a feature steered a model's prediction(s) by removing each feature and seeing the effect on the error.
# 
# - SHAP has:
#     - `TreeExplainer`:
#         - compatible with sckit learn, xgboost, Catboost
#     - `KernelExplainer`:
#         - compatible with "any" model
#         
# 
# 
# - See [this blog post](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d) for intro to topic and how to use with trees
# 
# - For non-tree/random forest models [see this follow up post]( https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a)
# 
#         

# 
# - Import and initialize javascript:
# 
# ```python
# import shap 
# shap.initjs()
# ```

# In[50]:


# !pip install shap


# In[51]:


import shap 
print(shap.__version__)
shap.initjs()


# In[52]:


rf = RandomForestClassifier()
pf.fit_and_time_model(rf,*train_test_list)


# ### To Get Expanations for Trees:
# 

# 
# 1. Create a shap explainer using your fit model.
# 
# ```python
# explainer = shap.TreeExplainer(xgb_clf)
# ```
# 
# 2. Get shapely values from explainer for your training data
# 
# ```python
# shap_values = explainer.shap_values(X_train,y_train)
# ```            
# 
# 3. Select which type of the available plots you'd like to visualize
# 
#     
# - **Types of Plots:**
#     - `summary_plot()`
#     - `dependence_plot()`
#     - `force_plot()` for a given observation
#     - `force_plot()` for all data
#     
#   

# In[53]:


# importances = pf.get_importance(rf,X_train_df,top_n=30)


# In[54]:


## Initialize an explainer with the model
explainer = shap.TreeExplainer(rf)

## Calculaate shap values for test data
shap_values = explainer.shap_values(X_test_df,y_test)
len(shap_values)


# In[55]:


shap_values[1].shape, X_test_df.shape


# ### Summary Plot

# ```python
# ## For normal bar graph of importance:
# shap.summary_plot(shap_values[1],X_train,plot_type='bar')
# 
# ## For detail Shapely value visuals:
# shap.summary_plot(shap_values, X_train)
# ```
#   
# 
# **`shap.summary_plot`**
# > - Feature importance: Variables are ranked in descending order.
# - Impact: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.
# - Original value: Color shows whether that variable is high (in red) or low (in blue) for that observation.
# 
# 
# >- **IMPORTANT NOTE:** You may need to slice out the correct shap_values for the target class. (by default explainer.shap_values seems to return a list for a binary classification, one set of shap values for each class).
#     - This will cause issues like the summary plot having a bar with an equal amount of blue and red for each class. 
#     - To fix, slice out the correct matrix from shap_values [0,1]
# 

# In[56]:


shap.summary_plot(shap_values[1], X_test_df,plot_type='bar',max_display=40)


# In[57]:


shap.summary_plot(shap_values[1],X_test_df,max_display=40)


# In[ ]:





# ### Force Plots

# `shap.force_plot`
# 
# To show an individual data point's prediction and the factors pushing it towards one class or another
# 
# ```python
# ## Just using np to randomly select a row
# row = np.random.choice(range(len(X_train)))
#                        
# shap.force_plot(explainer.expected_value, shap_values[row,:], X_train.iloc[row,:])
# ```

# #### Explain Individual Plot

# In[58]:


target_lookup = {0:'Non-User',1:'Heroin User'}
target_lookup[0]


# In[59]:


row = np.random.choice(range(len(X_test_df)))
print(f"- Row #: {row}")
print(f"Class = {target_lookup[y_test.iloc[row]]}")
X_test_df.iloc[row].round(2)


# In[60]:


## Individual forceplot
shap.force_plot(explainer.expected_value[1], shap_values[1][row],X_test_df.iloc[row])       


# In[61]:


## Overall Forceplot
shap.force_plot(explainer.expected_value[1], shap_values[1],X_test_df)       


# **`shap.dependence_plot`**

# 
# ```python
# ## To Auto-Select Feature Most correlated with a specific feature, just pass the desired feature's column name.
# 
# shap.dependence_plot('super_dist', shap_values, X_train)
# 
# ## There is a way to specifically call out multiple features but I wasn't able to summarize it quickly for this nb
# ```

# In[64]:


X_test_df


# In[65]:


shap.dependence_plot('Education',shap_values[1],X_test_df,
                     interaction_index='Age')


# ### Using SHAP with SVMS: 

# https://slundberg.github.io/shap/notebooks/Iris%20classification%20with%20scikit-learn.html**
# - ~~Must run the SVC with `probability=True` to be able to use `.predict_proba`, which is needed for the `KernelExplainer`~~

# In[66]:


svc_linear = pf.fit_and_time_model(SVC(kernel='linear',C=1),#probability=True,
                                *train_test_list)


# In[67]:


# pred_func = svc_linear.decision_function
X_shap = shap.sample(X_test_df,nsamples=200)
explainer = shap.KernelExplainer(svc_linear.predict,X_shap)
explainer


# In[68]:


shap_values = explainer.shap_values(X_shap,nsamples=100)#, nsamples=1000)


# In[69]:


shap_values.shape


# In[70]:


X_test_df.shape


# In[71]:


shap_values[0].shape


# In[ ]:





# In[72]:


# # shap.force_plot(shap_values[0],X_test)
shap.summary_plot(shap_values,X_shap) 


# In[73]:


# shap.summary_plot(shap_values,X_test,plot_type='bar')


# ## Explaining Models with LIME

# >- LIME (Local Interpretable Model-Agnostic Explanations) 
#     - GitHub: https://github.com/marcotcr/lime
#     - [White Paper](https://arxiv.org/abs/1602.04938)
# 
# - [Blog Post:"ExplainYour Modelw ith LIME"](https://medium.com/dataman-in-ai/explain-your-model-with-lime-5a1a5867b423)

# In[74]:


# !pip install lime
from lime.lime_tabular import LimeTabularExplainer


# In[75]:


lime_explainer =LimeTabularExplainer(
    training_data=np.array(X_test_df),
    feature_names=X_train_df.columns,
    class_names=['Non-User', 'Heroin-User'],
    mode='classification'
)


# In[76]:


row = np.random.choice(range(len(X_test_df)))
print(f"- Row #: {row}")
print(f"Class = {target_lookup[y_test.iloc[row]]}")
# X_test_df.iloc[row].round(2)    


# In[77]:


exp = lime_explainer.explain_instance(X_test_df.iloc[row], rf.predict_proba)
exp.show_in_notebook(show_table=True)


# # Appendix

# ### Renaming Features

# In[78]:


columns= {0:'ID',1:'Age',2:'Gender',3:'Education',4:'Country',5:'Ethnicity',
         6:'Nscore',7:'Escore',8:'Oscore',9:"Ascore",10:'Cscore',
         11:'Impulsiveness',12:'SS'}

drug_dict = {13:'Alcohol', 
              14: 'Amphet', 
              15: 'Amyl', 
              16: 'Benzos', 
              17: 'Caff', 
              18: 'Cannabis', 
              19: 'Choc', 
              20: 'Coke',
              21: 'Crack', 
              22: 'Ecstasy',
              23: 'Heroin', 
              24: 'Ketamine', 
              25: 'LegalH', 
              26: 'LSD',
              27: 'Meth',
              28: 'Mushrooms', 
              29: 'Nicotine', 
              30: "FakeDrugSemeron",
              31: 'VSA'}

all_columns = {**columns, **drug_dict}

drug_cols = list(drug_dict.values())
drug_cols


# In[79]:


education, age, country,


# In[ ]:




