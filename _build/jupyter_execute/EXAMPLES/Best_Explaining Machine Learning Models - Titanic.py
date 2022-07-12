#!/usr/bin/env python
# coding: utf-8

# # Explaining Machine Learning Models

# ## Overview

# - Fit an LogisticRegression and RandomForest for the titanic dataset.
# 
# - Discuss 3 different methods for interpreting the models' results and what features it used to make its predictions. 
#     - Permutation Importance
#     - Using `SHAP` and Shapely Values
#     - Using `LIME`

# ## Notebook Headers for Study Group

# - ⭐️**EXPLAINING MACHINE LEARNING MODELS**⭐️
#     - All of today's content.
# -  📚**Shap Resources**
#     - Collection of Videos, Book Excerpts, and Blogs.

# ___

# # Imports and Functions

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


# !pip install tzlocal


# In[5]:


## Adding folder above to path
import os, sys
sys.path.append(os.path.abspath('../'))

## Load stack_functions with autoreload turned on
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from CODE import project_functions as pf

def show_code(function):
    import inspect 
    from IPython.display import display,Markdown, display_markdown
    code = inspect.getsource(function)
    md_txt = f"```python\n{code}\n```"
    return display(Markdown(md_txt))


# In[6]:


# %load_ext autoreload
# %autoreload 2



# ## Preprocessing

# In[7]:


# df = pd.read_csv('https://raw.githubusercontent.com/jirvingphd/dsc-phase-3-project/main/data/drug_use_renamed_converted.csv')#'data/drug_use_renamed_converted.csv')
# df


# In[8]:


fname = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vR9Yhcch85ziIad4CyZQqLtYijHgsuENLfyC0YAXlajVbSC7U7s3CUUsfG-OYIKOjTI9DdKZH1xMs3a/pub?output=csv'
# df.to_csv(fname,compression='gzip',index=False)
df= pd.read_csv(fname,index_col=0)
df


# In[9]:


df.info()


# In[10]:


df['Name'].apply(lambda x: x.split(',')[0]).value_counts()#.head(50)


# ### Identifying Columns for Preprocessing

# In[11]:


## dropping column that are not helpful
bad_cols = ['Name','Ticket']
df.drop(columns=bad_cols,inplace=True)


# In[12]:


object_cols = list(df.drop(columns='Survived').select_dtypes('object').columns)
object_cols


# In[13]:


df['Embarked'].value_counts()


# In[14]:


df['Sex'].value_counts()


# In[15]:


df['Cabin'].value_counts()


# ## Feature Engineering

# In[16]:


#?You 


# # Preprocessing

# In[17]:


df


# In[18]:


## Specifying root names of types of features to loop through and filter out from df
target_col = 'Survived'
drop_cols = ['Cabin']

y = df[target_col].copy()
X = df.drop(columns=[target_col,*drop_cols]).copy()
y.value_counts(1,dropna=False)


# In[19]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=321)
X_train


# In[20]:


from sklearn import set_config
set_config(display='diagram')


# In[21]:


## saving list of numeric vs categorical feature
num_cols = list(X_train.select_dtypes('number').columns)
cat_cols = list(X_train.select_dtypes('object').columns)

## create pipelines and column transformer
num_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scale',StandardScaler())#MinMaxScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='constant',fill_value='MISSING')),
    ('encoder',OneHotEncoder(sparse=False,handle_unknown='ignore',
                             drop='first')#,
#                              min_frequency=2)
    )])

print('# of num_cols:',len(num_cols))
print('# of cat_cols:',len(cat_cols))


# In[22]:


## COMBINE BOTH PIPELINES INTO ONE WITH COLUMN TRANSFORMER
preprocessor=ColumnTransformer(transformers=[
    ('num',num_transformer,num_cols),
    ('cat',cat_transformer,cat_cols)], verbose_feature_names_out=False)

preprocessor


# In[23]:


## Fit preprocessing pipeline on training data and pull out the feature names and X_cols
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()
## Transform X_traian,X_test and remake dfs
X_train_df = pd.DataFrame(preprocessor.transform(X_train),
                          index=X_train.index, columns=feature_names)
X_test_df = pd.DataFrame(preprocessor.transform(X_test),
                          index=X_test.index, columns=feature_names)
X_train_df


# In[24]:


# ## Fit preprocessing pipeline on training data and pull out the feature names and X_cols
# preprocessor.fit(X_train)

# ## Use the encoder's .get_feature_names
# cat_features = list(preprocessor.named_transformers_['cat'].named_steps['encoder']\
#                             .get_feature_names(cat_cols))
# X_cols = num_cols+cat_features


# In[25]:


# ## Transform X_traian,X_test and remake dfs
# X_train_df = pd.DataFrame(preprocessor.transform(X_train),
#                           index=X_train.index, columns=X_cols)
# X_test_df = pd.DataFrame(preprocessor.transform(X_test),
#                           index=X_test.index, columns=X_cols)

# ## Tranform X_train and X_test and make into DataFrames
# X_train_df


# In[26]:


y.value_counts(1)


# ## Resampling with SMOTENC

# In[27]:


# y_train.value_counts(1)


# In[28]:


# ## Save list of trues and falses for each cols
# smote_feats = [False]*len(num_cols) +[True]*len(cat_features)
# # smote_feats


# In[29]:


# ## resample training data
# smote = SMOTENC(smote_feats)
# X_train_sm,y_train_sm = smote.fit_resample(X_train_df,y_train)
# y_train_sm.value_counts()


# # MODELING

# #### Setting `train_test_list`

# In[30]:


### SAVING XY DATA TO LIST TO UNPACK
train_test_list = [X_train_df,y_train, X_test_df,y_test]


# ## Linear SVC

# In[31]:


logreg = LogisticRegression()#kernel='linear',C=1)
pf.fit_and_time_model(logreg,*train_test_list)


# ## RandomForest

# In[32]:


rf = RandomForestClassifier()
pf.fit_and_time_model(rf,*train_test_list)


# In[33]:


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

# In[34]:


from sklearn.inspection import permutation_importance


# In[35]:


## Permutation importance takes a fit mode and test data. 
r = permutation_importance(logreg, X_test_df, y_test,
                           n_repeats=30,scoring='f1')
r.keys()


# In[36]:


r['importances_mean']


# In[37]:


## can make the mean importances into a series
permutation_importances = pd.Series(r['importances_mean'],index=X_train_df.columns,
                           name = 'permutation importance')
permutation_importances


# In[38]:


r = permutation_importance(rf, X_test_df, y_test, n_repeats=30, scoring='f1')
rf_importances = pd.Series(r['importances_mean'],index=X_test_df.columns,
                          name= 'rf permutation importance')
rf_importances


# In[39]:


embedded_importances = pf.get_importance(rf,X_test_df)#,plot=False)
embedded_importances.name ='rf.feature_importances_'
embedded_importances


# In[40]:


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


# In[41]:


## Compare embedded feature importance vs permutation importance
comp = compare_importances(embedded_importances,rf_importances,#svc_importances,
                          show_bar=True,sort_index=False,sort_col=0)
comp


# In[42]:


# df['']


# In[43]:


## Compare embedded feature importance vs permutation importance
comp = compare_importances(embedded_importances,permutation_importances,
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

# In[44]:


# !pip install shap


# In[45]:


import shap 
print(shap.__version__)
shap.initjs()


# In[46]:


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

# In[47]:


# importances = pf.get_importance(rf,X_train_df,top_n=30)


# In[48]:


## Initialize an explainer with the model
explainer = shap.TreeExplainer(rf)

## Calculaate shap values for test data
shap_values = explainer.shap_values(X_test_df,y_test)
len(shap_values)


# In[49]:


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

# In[50]:


shap.summary_plot(shap_values[1], X_test_df,plot_type='bar',max_display=40)


# In[51]:


shap.summary_plot(shap_values[1],X_test_df,max_display=40)


# In[ ]:





# # Bookmark 06/03/22

# ### Force Plots

# `shap.force_plot`

# To show an individual data point's prediction and the factors pushing it towards one class or another

# ```python
# ## Just using np to randomly select a row
# row = np.random.choice(range(len(X_train)))
#                        
# shap.force_plot(explainer.expected_value, shap_values[row,:], X_train.iloc[row,:])
# ```

# #### Explain Individual Plot

# In[52]:


target_lookup = {0:'Died',1:'Survived'}
target_lookup[0]


# In[53]:


## sandomly select a sample of 1 from test data
row = np.random.choice(range(len(X_test_df)))
print(f"- Row #: {row}")

## Get y-value for selected row
class_ = target_lookup[y_test.iloc[row]]
print(f"- Class = {class_}")
sample = X_test_df.iloc[row]#.round(2)
sample


# In[54]:


sample.loc[num_cols]


# In[55]:


## reshaping numeric cols so can inverse transform
sample_num = sample.loc[num_cols].values.reshape(1,-1)


# In[56]:


##  Our scaler
scaler = preprocessor.named_transformers_['num'][-1]
scaler


# In[57]:


## Inverse transform and make into a datafarme
sample_inv = pd.DataFrame(scaler.inverse_transform(sample_num),
          columns=num_cols,index=[row])
sample_inv.insert(0,'Target',class_)
sample_inv


# In[58]:


##  Our encoder
encoder = preprocessor.named_transformers_['cat'][-1]
encoder


# In[59]:


## Getting final OHE column names
cat_features = encoder.get_feature_names_out(cat_cols)
cat_features


# In[60]:


sample[cat_features].to_frame(row).T


# In[61]:


## Concat with the categorical features
sample_df = pd.concat([sample_inv,sample[cat_features].to_frame(row).T],axis=1)
sample_df


# In[62]:


## Individual forceplot
shap.force_plot(explainer.expected_value[1], shap_values[1][row],X_test_df.iloc[row])       


# In[63]:


## Trying to find num_cols and cat_cols in the preprocessor
params = preprocessor.get_params()
[print(f"- {k}") for k in params.keys()];


# In[64]:


## slice out first transformer pipeline
preprocessor.transformers_[0]


# In[65]:


## list of features is final element in pipeline
preprocessor.transformers_[0][-1]


# In[66]:


preprocessor.transformers_


# In[67]:


# pararms


# In[ ]:





# In[68]:


# params['cat']t


# In[69]:


## testing function code
num_pipe_name = 'num'
num_pipe_idx = [i for i,pipe in enumerate(preprocessor.transformers_) if pipe[0]==num_pipe_name]
num_pipe_idx


# In[70]:


num_pipe = preprocessor.transformers_[num_pipe_idx[0]]
num_pipe[-1]


# In[71]:


preprocessor.named_transformers_['num']


# In[72]:


sample.to_frame().T


# In[73]:


## Putting it all together

def get_sample_row(X_test_df,y_test, preprocessor, row=None, inverse_tf = True,
                   target_lookup = {0:'Died',1:'Survived'}, 
                   num_pipe_name = 'num', cat_pipe_name='cat', 
                   use_orig_index = False, class_pos = -1,
                  as_index=False):
    """Extracts a single row as a new dataframe to compare to shap force plot"""
    
    if row is None:
        if use_orig_index==False:
            ## sandomly select a sample of 1 from test data
            row = np.random.choice(range(len(X_test_df)))
            ## Get y-value for selected row
            class_ = target_lookup[y_test.iloc[row]]
            sample = X_test_df.iloc[row]
            ## save final index name
            row_num = row
            idx_name ='row #'
        else: 
            row = np.random.choice(X_test_df.index)
            ## Get y-value for selected index
            class_ = target_lookup[y_test.loc[row]]
            sample = X_test_df.loc[row]
            row_num = X_test_df.index.get_loc(row)
            idx_name = 'idx'
            
    print(f"- Row #: {row}")
    print(f"- Class = {class_}")


    if inverse_tf==True:
        ## Saving numeric pipeline vars

        ## find the numeric transformer in preprocessor.TRANSFORMERS_ (not named)
        tfs = preprocessor.transformers_
        num_pipe_idx = [i for i,pipe in enumerate(tfs) if pipe[0]==num_pipe_name]
        num_pipe = preprocessor.transformers_[num_pipe_idx[0]]

        ## Grabbing the last item in the actual pipeline (index=1)
        scaler = num_pipe[1][-1]
        num_cols = num_pipe[-1]

        ## Get the individual sample's num values to inv_tf
        sample_num = sample.loc[num_cols].values.reshape(1,-1)


        ## Inverse transform and make into a datafarme
        sample_inv = pd.DataFrame(scaler.inverse_transform(sample_num),
                  columns=num_cols,index=[row])
        sample_inv.insert(0,'Target',class_)

        ## Saving needed info from cat pipeline
        cat_pipe_idx = [i for i,pipe in enumerate(tfs) if pipe[0]==cat_pipe_name]
        cat_pipe = preprocessor.transformers_[cat_pipe_idx[0]]

        encoder = cat_pipe[1][-1]
        cat_cols = cat_pipe[-1]
        cat_features = encoder.get_feature_names_out(cat_cols)


        ## Inverse transform and make into a dataframe
        sample_inv = pd.DataFrame(scaler.inverse_transform(sample_num),
                  columns=num_cols,index=[row])
            ## Concat with the categorical features
        sample_df = pd.concat([sample_inv,
                               sample[cat_features].to_frame(row).T],axis=1)
    else:
        sample_df = sample.to_frame().T
        
    ## insert class and add index name
    if class_pos==-1:
        class_pos = len(sample_df.columns)
    sample_df.insert(class_pos,'Target',class_)
    
    ## Insert index/row
    sample_df.index.name = idx_name
    
    if use_orig_index==True:
        sample_df.insert(0,'row #',row_num)
    else:
        sample_df.insert(0,'idx',X_test_df.index[row_num])

    if as_index:
        return sample_df
    else:
        return sample_df.reset_index()


# In[74]:


sample_df = get_sample_row(X_test_df, y_test,preprocessor, use_orig_index=True)
sample_df


# In[75]:


sample_df = get_sample_row(X_test_df, y_test,preprocessor, use_orig_index=False)
sample_df


# In[76]:


## Individual forceplot
shap.force_plot(explainer.expected_value[1], shap_values[1][sample_df['row #']],X_test_df.iloc[sample_df['row #']])       


# In[77]:


def get_sample_show_forceplot(X_test_df,y_test,preprocessor,  row=None, 
                              inverse_tf = True, 
                              target_lookup = {0:'Died',1:'Survived'}, 
                              num_pipe_name = 'num', cat_pipe_name='cat', 
                              use_orig_index = False, class_pos = -1,
                              as_index=False):
    sample_df = get_sample_row(X_test_df, y_test,preprocessor, row=row, 
                               inverse_tf=inverse_tf,target_lookup=target_lookup,
                               num_pipe_name=num_pipe_name, 
                               cat_pipe_name=cat_pipe_name, 
                               use_orig_index=use_orig_index, class_pos=class_pos,
                              as_index=as_index)
    ## Individual forceplot
    display(sample_df)
    return shap.force_plot(explainer.expected_value[1], shap_values[1][sample_df['row #']],X_test_df.iloc[sample_df['row #']])       


# In[78]:


get_sample_show_forceplot(X_test_df,y_test, preprocessor)


# ### Overall Forceplot

# In[ ]:





# In[79]:


## Overall Forceplot
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test_df)       


# **`shap.dependence_plot`**

# 
# ```python
# ## To Auto-Select Feature Most correlated with a specific feature, just pass the desired feature's column name.
# 
# shap.dependence_plot('super_dist', shap_values, X_train)
# 
# ## There is a way to specifically call out multiple features but I wasn't able to summarize it quickly for this nb
# ```

# In[80]:


X_test_df.columns


# In[81]:


## ERROR WHEN RUNNING DEPENDENCE_PLOT WITHOUT EXPLICITY itneraction_index
## https://stackoverflow.com/questions/70482847/shap-dependence-plot-doesnt-work-with-an-error-which-is-related-deep-inside-th
try:
    shap.dependence_plot('Sex_male',shap_values[1],X_test_df,
                         interaction_index='Age')
except Exception as e:
    print("ERROR! ",e)


# - Workaround: manually determining the ideal interaction_index the same way as the function.
# `shap.common.approximate_interactions`
# 
# From the documentation:
# > - interaction_index : "auto", None, int, or string
#     - The index of the feature used to color the plot. The name of a feature can also be passed as a string. 
#     - If "auto" then shap.common.approximate_interactions is used to pick what seems to be the strongest interaction (note that to find to true stongest interaction you need to compute the SHAP interaction values).

# In[ ]:





# In[82]:


import inspect
from IPython.display import Markdown

md = "```python\n"+inspect.getsource(shap.approximate_interactions)+"\n```"
Markdown(md)


# In[83]:


## manually calculating the strongest feature interactions
# to use as interaction_index for shap.dependence_plot
interactions = shap.approximate_interactions('Age',shap_values[1],X_test_df,
                              feature_names=X_test_df.columns)
interactions


# In[84]:


X_test_df.columns


# In[85]:


X_test_df.columns[interactions]


# In[86]:


## save the results in a datframe and add rank column to help sift through
strong_intxn = pd.DataFrame({ "Feature": X_test_df.columns[interactions],
                             'Interaction Index':interactions,
                             'Rank': range(1,len(interactions)+1) ,
                           })
strong_intxn = strong_intxn.set_index('Feature')
strong_intxn


# In[87]:


## what is the smallest value for interaction index
print(strong_intxn['Interaction Index'].idxmin())
strong_intxn['Interaction Index'].idxmax()


# In[88]:


shap.dependence_plot('Age',shap_values[1], features= X_test_df,
                     feature_names=X_test_df.columns,
                     interaction_index= strong_intxn['Interaction Index'].idxmin())


# In[89]:


corr_age = X_test_df.corrwith(X_test_df['Age']).to_frame('Corr')
corr_age


# In[90]:


corr_age['AbsCorr'] = corr_age['Corr'].abs()


# In[91]:


sns.heatmap(corr_age.sort_values('AbsCorr',ascending=False), annot=True, cmap='coolwarm_r')


# In[92]:


strong_intxn['Interaction Index'].idxmin()


# In[93]:


shap.dependence_plot('Age',shap_values[1], features= X_test_df,
                     feature_names=X_test_df.columns,
                     interaction_index='Pclass')


# In[94]:


shap.dependence_plot('Age',shap_values[1], features= X_test_df,
                     feature_names=X_test_df.columns,
                     interaction_index='SibSp')


# In[95]:


# shap.dependence_plot('Age',shap_values[1], features= X_test_df,
#                      feature_names=X_test_df.columns,
#                      interaction_index= strong_intxn['Interaction Index'].idxmax())


# In[ ]:





# ### Trying to Understand the Interaction Values Better (and If I am using them)

# - https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a

# In[96]:


interaction_vals = explainer.shap_interaction_values(X_test_df)
type(interaction_vals)


# In[97]:


len(interaction_vals)


# In[98]:


interaction_vals[1].shape


# In[99]:


X_test_df.shape


# In[100]:


shap.summary_plot(interaction_vals[1], X_test_df)


# In[101]:


interaction_vals[1].shape


# In[102]:


interaction_vals[1][0].shape


# In[103]:


# # Source: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html
# """
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(Xd)
# np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
# """

# np.abs(shap_values[1].sum(1) + explainer.expected_value - rf.predict(X_test_df)).max()


# In[104]:


intxn_vals_survived = interaction_vals[1]
intxn_vals_survived 


# In[ ]:





# In[105]:


pd.DataFrame(intxn_vals_survived.mean(axis=0),  # axis=1?
             index=X_test_df.columns,
             columns=X_test_df.columns)


# In[106]:


# shap.dependence_plot('Age',shap_values[1], features= X_test_df,
#                      feature_names=X_test_df.columns,
#                      interaction_index= strong_intxn['Interaction Index'].idxmax())


# ### Waterfall Plot

# In[107]:


# explainer.expected_value


# In[108]:


# shap_values


# In[109]:


#source: https://towardsdatascience.com/explainable-ai-xai-a-guide-to-7-packages-in-python-to-explain-your-models-932967f0634b
i = 79
## MAKE SURE TO SLICE OUT [1] FROM shap_values and expected_value before slicing out the row
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][i], 
                                       features=X_test_df.iloc[i,:], 
                                       feature_names=X_test_df.columns, 
                                       max_display=15, show=True)


# ## Testing interpretml

# In[110]:


# from sklearn.model_selection import train_test_split
# from interpret.glassbox import ExplainableBoostingClassifier
# #the glass box model (using Boosting Classifier)
# ebm = ExplainableBoostingClassifier(random_state=120)
# # titanic = sns.load_dataset('titanic').dropna()
# # #Data splitting
# # X_train, X_test, y_train, y_test = train_test_split(titanic.drop(['survived', 'alive'], axis = 1), 
# #                                                     titanic['survived'], test_size = 0.2, random_state = 121)
# #Model Training
# ebm.fit(X_train_df, y_train)


# In[111]:


# shap.summary_plot(shap_values,X_test,plot_type='bar')


# ## Explaining Models with LIME

# >- LIME (Local Interpretable Model-Agnostic Explanations) 
#     - GitHub: https://github.com/marcotcr/lime
#     - [White Paper](https://arxiv.org/abs/1602.04938)
# 
# - [Blog Post:"ExplainYour Modelw ith LIME"](https://medium.com/dataman-in-ai/explain-your-model-with-lime-5a1a5867b423)

# In[112]:


# !pip install lime
from lime.lime_tabular import LimeTabularExplainer


# In[113]:


lime_explainer =LimeTabularExplainer(
    training_data=np.array(X_test_df),
    feature_names=X_train_df.columns,
    class_names=['Died', 'Survived'],
    mode='classification'
)


# In[114]:


row = np.random.choice(range(len(X_test_df)))
print(f"- Row #: {row}")
print(f"Class = {target_lookup[y_test.iloc[row]]}")
# X_test_df.iloc[row].round(2)    


# In[115]:


exp = lime_explainer.explain_instance(X_test_df.iloc[row], rf.predict_proba)
exp.show_in_notebook(show_table=True)


# In[116]:


from ipywidgets import interact

@interact
def show_instance(row=(0, len(X_test_df))):
    print(f"- Row #: {row}")
    print(f"Class = {target_lookup[y_test.iloc[row]]}")
    exp = lime_explainer.explain_instance(X_test_df.iloc[row], rf.predict_proba)

    exp.show_in_notebook(show_table=True)
# X_test_df.iloc[row].round(2)    


# In[ ]:




