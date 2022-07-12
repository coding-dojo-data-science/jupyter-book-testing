#!/usr/bin/env python
# coding: utf-8

# # Template: Preview Regression Dataset

# - Goal of this notebook is to paste a template version of the workflow from the ADMIN_Comparing_Datasets notebook.
#  
# **The code will:**
# - Create 4 preprocessing pipelines (one with scaling and one without)
#     - `preprocessor`
#         - Clone: `preprocessor_cln`
#     - `preprocessor_scale`
#         - Clone: `preprocessor_scale_cln`    
#     
# - Create 4 Versions of the X/y data.
#     - **All Data/Rows:**
#         - Starting Vars:
#             - df,X,y, X_train, X_test,y_train,y_test
#         - **Unscaled**
#             - Without a  constant:
#                 - X_train_df, X_test_df, y_train,y_test
#             - With a constant:
#                 - X_train_df_cnst, X_test_df_cnst, y_train (same), y_test (same)
#         - **Scaled:**
#             - Without a  constant:
#                 - X_train_df_scaled, X_test_df_scaled, y_train (same), y_test (same)
#             - With a constant:
#                 - X_train_df_scaled_cnst, X_test_df_scaled_cnst, y_train (same), y_test (same)
# 
#     - **Cleaned/Outliers Removed**
#         - Starting Vars:
#             - df_clean,X_cln,y_cln, X_train_cln, X_test_cln,y_train_cln,y_test_cln
#         - **Unscaled**
#             - Without a  constant:
#                 - X_train_df_cln, X_test_df_cln, y_train,y_test
#             - With a constant:
#                 - X_train_df_cln_cnst, X_test_df_cln_cnst, y_train_cln (same), y_test_cln (same)
#         - **Scaled:**
#             - Without a  constant:
#                 - X_train_df_cln_scaled, X_test_df_cln_scaled, y_train_cln (same), y_test_cln (same)
#             - With a constant:
#                 - X_train_df_scaled_cln_cnst, X_test_df_cln_scaled_cnst, y_train_cln (same), y_test_cln (same)

# # Code

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
## Customization Options
# pd.set_option('display.float_format',lambda x: f"{x:,.4f}")
pd.set_option("display.max_columns",100)
plt.style.use(['fivethirtyeight','seaborn-talk'])
mpl.rcParams['figure.facecolor']='white'

## additional required imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import metrics

from sklearn.base import clone

## fixing random for lesson generation
np.random.seed(321)

##import statsmodels correctly
import statsmodels.api as sm
from scipy import stats

## Customized Options
# pd.set_option('display.float_format',lambda x: f"{x:,.4f}")
# plt.style.use('seaborn-talk')


# In[ ]:





# In[2]:


## Adding folder above to path
import os, sys
sys.path.append(os.path.abspath('../'))

## Load stack_functions with autoreload turned on
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from CODE import stack_functions as sf

def show_code(function):
    import inspect 
    from IPython.display import display,Markdown, display_markdown
    code = inspect.getsource(function)
    md_txt = f"```python\n{code}\n```"
    return display(Markdown(md_txt))


# ## Preliminary Checks and Dtype Conversion

# - Change:
#     - `FILE`: url or filepath to load
#     - `DROP_COLS`: list of columns to drop from df
#     - `CONVERT_TO_STR_COLS`: numeric cols to convert to str
#     - `CONVERT_TO_NUM_COLS`: str cols to convert to numeric (uses pd.to_numeric)

# In[ ]:


## Load in data
FILE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSEZQEzxja7Hmj5tr5nc52QqBvFQdCAGb52e1FRK1PDT2_TQrS6rY_TR9tjZjKaMbCy1m5217sVmI5q/pub?output=csv"
df = pd.read_csv(FILE)


# In[ ]:


from pandas_profiling import ProfileReport
ProfileReport(df)


# In[ ]:


df.info()
df.head()


# In[ ]:


## Drop unwanted columns
DROP_COLS = []#'id','date']
df = df.drop(columns=DROP_COLS)


# In[ ]:


## Convert all categories to strings
CONVERT_TO_STR_COLS = []#'zipcode']
df[CONVERT_TO_STR_COLS] = df[CONVERT_TO_STR_COLS].astype(str)

CONVERT_TO_NUM_COLS = []
for col in CONVERT_TO_NUM_COLS:
    df[col] = pd.to_numeric(df[col])


# In[ ]:


## final info before X/y
df.info()


# ## Full Dataset Preprocessing

# In[ ]:


## Make x and y variables
target = None#'price'
drop_cols_model = []

y = df[target].copy()
X = df.drop(columns=[target,*drop_cols_model]).copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=321)
X_train.head()


# ### Unscaled

# In[ ]:


## cat selector
cat_sel = make_column_selector(dtype_include='object')
cat_cols = cat_sel(X)

# num selectorr
num_sel = make_column_selector(dtype_include='number')
num_cols = num_sel(X)

## make pipelines & column transformer - raw numeric
cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
                                       fill_value='MISSING'),
                         OneHotEncoder(handle_unknown='ignore', sparse=False))

num_pipe_raw = make_pipeline(SimpleImputer(strategy='mean'))
preprocessor = make_column_transformer((num_pipe_raw, num_sel),
                                       (cat_pipe,cat_sel), verbose_feature_names_out=False)
preprocessor


# In[ ]:


### PREP ALL X VARS
## Prepare X_train_df
X_train_df = pd.DataFrame( preprocessor.fit_transform(X_train), 
                          columns=preprocessor.get_feature_names_out(),
                         index=X_train.index)

## Prepare X_test_df
X_test_df = pd.DataFrame( preprocessor.transform(X_test),
                         columns=preprocessor.get_feature_names_out(), 
                         index=X_test.index)


## Prepare X vars with constant
X_train_df_cnst = sm.add_constant(X_train_df, prepend=False, has_constant='add')
X_test_df_cnst = sm.add_constant(X_test_df, prepend=False, has_constant='add')


# In[ ]:


X_train_df.describe()


# > The cell below should be updated for each dataset- only separating high cardinality OHE features (e.g. zipcode)

# In[ ]:


## Save list of zipcode columns and other columns
ohe_cols = [c for c in X_train_df.columns if c.startswith('zipcode')]
nonohe_cols = X_train_df.drop(columns=[*ohe_cols]).columns.tolist()


# ### Scaled

# In[ ]:


## make pipelines & column transformer - scaled
cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
                                       fill_value='MISSING'),
                         OneHotEncoder(handle_unknown='ignore', sparse=False))
num_pipe_scale = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
preprocessor_scale = make_column_transformer((num_pipe_scale, num_sel),
                                       (cat_pipe,cat_sel), verbose_feature_names_out=False)
preprocessor_scale


# In[ ]:


# ### PREP ALL SCALED X VARS


# Prepare X_train_scaled_df & X_test_scaled_df
X_train_scaled_df = pd.DataFrame( preprocessor_scale.fit_transform(X_train), 
                          columns=preprocessor_scale.get_feature_names_out(),
                         index=X_train.index)


X_test_scaled_df = pd.DataFrame( preprocessor_scale.transform(X_test),
                         columns=preprocessor_scale.get_feature_names_out(), 
                         index=X_test.index)


## Save vers with constant
X_train_scaled_df_cnst = sm.add_constant(X_train_scaled_df, prepend=False, has_constant='add')
X_test_scaled_df_cnst = sm.add_constant(X_test_scaled_df, prepend=False, has_constant='add')


## Save list of zipcode columns and other columns
ohe_cols = [c for c in X_train_df.columns if c.startswith('zipcode')]
nonohe_cols = X_train_df.drop(columns=[*ohe_cols]).columns.tolist()


X_test_scaled_df.head()


# In[ ]:


X_train_scaled_df.describe()


# ## Cleaned Dataset Preprocessing

# ### Remove Outliers

# In[ ]:


show_code(sf.remove_outliers)


# In[ ]:


df_clean_iqr = sf.remove_outliers(df,verbose=2)
df_clean_iqr


# In[ ]:


df_clean_z = sf.remove_outliers(df,method='z')
df_clean_z


# In[ ]:


## Make x and y variables
# target = 'price'
# drop_cols_model = []

y_cln = df_clean_z[target].copy()
X_cln = df_clean_z.drop(columns=[target,*drop_cols_model]).copy()

X_train_cln, X_test_cln, y_train_cln, y_test_cln = train_test_split(X_cln,y_cln, random_state=321)
X_train_cln.head()


# ### Unscaled

# In[ ]:


## Cloning the Previous 2 Preprocessors
preprocessor_cln = clone(preprocessor)
preprocessor_cln_scale = clone(preprocessor_scale)


# In[ ]:


### PREP ALL X VARS
## Prepare X_train_df
X_train_df_cln = pd.DataFrame( preprocessor_cln.fit_transform(X_train_cln), 
                          columns=preprocessor_cln.get_feature_names_out(),
                         index=X_train_cln.index)

## Prepare X_test_df
X_test_df_cln = pd.DataFrame( preprocessor_cln.transform(X_test_cln),
                         columns=preprocessor_cln.get_feature_names_out(), 
                         index=X_test_cln.index)


## Prepare X vars with constant
X_train_df_cln_cnst = sm.add_constant(X_train_df_cln, prepend=False, has_constant='add')
X_test_df_cln_cnst = sm.add_constant(X_test_df_cln, prepend=False, has_constant='add')


## Save list of zipcode columns and other columns
ohe_cols = [c for c in X_train_df_cln.columns if c.startswith('zipcode')]
nonohe_cols = X_train_df_cln.drop(columns=[*ohe_cols]).columns.tolist()

X_test_df.head()


# In[ ]:


X_train_df_cln.describe()


# ### Scaled

# In[ ]:


# ### PREP ALL SCALED X VARS


# Prepare X_train_scaled_df & X_test_scaled_df
X_train_scaled_df_cln = pd.DataFrame( preprocessor_cln_scale.fit_transform(X_train_cln), 
                          columns=preprocessor_cln_scale.get_feature_names_out(),
                         index=X_train_cln.index)


X_test_scaled_df_cln = pd.DataFrame( preprocessor_cln_scale.transform(X_test_cln),
                         columns=preprocessor_cln_scale.get_feature_names_out(), 
                         index=X_test_cln.index)


## Save vers with constant
X_train_scaled_df_cln_cnst = sm.add_constant(X_train_scaled_df_cln, prepend=False, has_constant='add')
X_test_scaled_df_cln_cnst = sm.add_constant(X_test_scaled_df_cln, prepend=False, has_constant='add')


## Save list of zipcode columns and other columns
ohe_cols = [c for c in X_train_scaled_df_cln.columns if c.startswith('zipcode')]
nonohe_cols = X_train_scaled_df_cln.drop(columns=[*ohe_cols]).columns.tolist()


X_train_scaled_df_cln.head()


# In[ ]:


X_train_scaled_df_cln.describe()


# ## Modeling - Full Dataset

# ### Raw Numeric - No Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model_raw = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result_raw = model_raw.fit()
sf.evaluate_ols(result_raw,X_train_df, y_train)


# In[ ]:


fig_raw = sf.plot_coeffs(result_raw, ohe_cols, include_const=True,title="Raw Coefficients")


# In[ ]:


fig_raw_zips =sf.plot_coeffs(result_raw, nonohe_cols, figsize=(6,12),
                           include_const=True,title="Raw Coefficients")


# ### Raw Numeric - with Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model_raw_cnst = sm.OLS(y_train, X_train_df_cnst)

## Fit the model and view the summary
result_raw_cnst = model_raw_cnst.fit()
sf.evaluate_ols(result_raw_cnst,X_train_df_cnst, y_train)


# In[ ]:


fig_raw =sf.plot_coeffs(result_raw_cnst, ohe_cols, include_const=True,title="Raw Coefficients")


# In[ ]:


fig_raw =sf.plot_coeffs(result_raw_cnst, ohe_cols, include_const=False,title="Raw Coefficients")


# In[ ]:


fig_raw_zips =sf.plot_coeffs(result_raw_cnst, nonohe_cols, figsize=(6,12),include_const=False,title="Raw Coefficients")


# In[ ]:





# ### Scaled Numeric - No Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model_scaled = sm.OLS(y_train, X_train_scaled_df)

## Fit the model and view the summary
result_scaled = model_scaled.fit()
sf.evaluate_ols(result_scaled,X_train_scaled_df, y_train)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_scaled,ohe_cols,include_const=True)


# In[ ]:


fig_scaled_zips =sf.plot_coeffs(result_scaled, nonohe_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# ### Scaled Numeric - with Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model_scaled_cnst = sm.OLS(y_train, X_train_scaled_df_cnst)

## Fit the model and view the summary
result_scaled_cnst = model_scaled_cnst.fit()
sf.evaluate_ols(result_scaled_cnst,X_train_scaled_df_cnst, y_train)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_scaled_cnst,ohe_cols,include_const=True)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_scaled_cnst,ohe_cols,include_const=False)


# In[ ]:


fig_scaled_zips =sf.plot_coeffs(result_scaled_cnst, nonohe_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# ## Modeling - No Outliers

# ### Raw Numeric - No Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model_cln = sm.OLS(y_train_cln, X_train_df_cln)

## Fit the model and view the summary
result_cln = model_cln.fit()
sf.evaluate_ols(result_cln,X_train_df_cln, y_train_cln)


# In[ ]:


fig_raw =sf.plot_coeffs(result_cln, ohe_cols, include_const=True,title="Raw Coefficients")


# In[ ]:


fig_raw_zips =sf.plot_coeffs(result_cln, nonohe_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# ### Raw Numeric - with Constant

# In[ ]:


X_train_df_cln_cnst.describe()


# In[ ]:


## instantiate an OLS model WITH the training data.
model_cln_cnst = sm.OLS(y_train_cln, X_train_df_cln_cnst)

## Fit the model and view the summary
result_cln_cnst = model_cln_cnst.fit()
sf.evaluate_ols(result_cln_cnst,X_train_df_cln_cnst, y_train_cln)


# In[ ]:


fig_raw =sf.plot_coeffs(result_cln_cnst, ohe_cols, include_const=True,title="Raw Coefficients")


# In[ ]:


fig_raw =sf.plot_coeffs(result_cln_cnst, ohe_cols, include_const=False,title="Raw Coefficients")


# In[ ]:


fig_raw_zips =sf.plot_coeffs(result_cln_cnst, nonohe_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Scaled Numeric - No Constant

# In[ ]:


X_train_scaled_df_cln.describe()


# In[ ]:


## instantiate an OLS model WITH the training data.
model_cln_scaled = sm.OLS(y_train_cln, X_train_scaled_df_cln)

## Fit the model and view the summary
result_cln_scaled = model_cln_scaled.fit()
sf.evaluate_ols(result_cln_scaled,X_train_scaled_df_cln, y_train_cln)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_cln_scaled,ohe_cols,include_const=True)


# In[ ]:


fig_scaled_zips =sf.plot_coeffs(result_cln_scaled, nonohe_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# ### Scaled Numeric - with Constant

# In[ ]:


X_train_scaled_df_cln_cnst.describe()


# In[ ]:


## instantiate an OLS model WITH the training data.
model_cln_scaled_cnst = sm.OLS(y_train_cln, X_train_scaled_df_cln_cnst)

## Fit the model and view the summary
result_cln_scaled_cnst = model_cln_scaled_cnst.fit()
sf.evaluate_ols(result_cln_scaled_cnst,X_train_scaled_df_cln_cnst, y_train_cln)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_cln_scaled_cnst,ohe_cols,include_const=True)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_cln_scaled_cnst,ohe_cols,include_const=False)


# In[ ]:





# In[ ]:





# # Adding Explanations

# ## Sklearn LinearRegression

# > Pick 1 of the statsmodels models above to remake in sklearn for model explanations

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


## Making new var name sfor sklearn - update these to change model
ols_results = result_scaled_cnst
# sf.evaluate_ols(ols_results,X_tr,y_tr)
X_tr = X_train_scaled_df_cnst
X_te = X_test_scaled_df_cnst
y_tr = y_train
y_te = y_test

## If const in orig df, 
fit_intercept = 'const' in X_tr.columns

if fit_intercept==True:
    X_tr = X_tr.drop(columns='const')
    X_te = X_te.drop(columns='const')
fit_intercept


# In[ ]:


linreg = LinearRegression(fit_intercept=fit_intercept)
linreg.fit(X_tr, y_tr)
print(f"Train R^2: {format(linreg.score(X_tr,y_tr),'.2f')}")
print(f"Test R^2: {format(linreg.score(X_te,y_te),'.2f')}")
linreg.get_params()


# In[ ]:


coeffs = sf.get_coeffs(linreg,X_tr,name='LinReg Coeffs')
coeffs#.head(20)


# In[ ]:


ax = coeffs.drop(ohe_cols).sort_values().plot(kind='barh',figsize=(4,6))
ax.axvline(0,color='black')


# In[ ]:


if len(ohe_cols)>1:

    ax = coeffs[ohe_cols].sort_values().plot(kind='barh',figsize=(4,6))
    ax.axvline(0,color='black')


# In[ ]:


## save 1 df of skelarn vs ols coeffs
compare_coeffs = pd.DataFrame({'OLS':ols_results.params,
                              'LinReg':coeffs}).round(2)
compare_coeffs['Agree?'] = compare_coeffs['OLS']==compare_coeffs['LinReg']

display(compare_coeffs.round(3))
compare_coeffs['Agree?'].value_counts(1)
#compare_coeffs.style.format({'OLS':"{:,.2f}","LinReg":"{:,.2f}"})


# In[ ]:


compare_coeffs[compare_coeffs['Agree?']==True]


# In[ ]:


# compare_coeffs[compare_coeffs['Agree?']==False]


# In[ ]:


compare_coeffs.style.bar()


# > ISSUE WITH COEFFICIENTS NOT MATCHING - SEE IF TRUE FOR OTHER DATASETS. 

# ### Shap

# In[ ]:


import shap
shap.initjs()

shap.__version__


# In[ ]:


## sampling 200 rows from training data
X_shap = shap.sample(X_tr,nsamples=200,random_state=321)


# In[ ]:


## Creating explainer from model and getting shap values
explainer = shap.LinearExplainer(linreg,X_shap)
shap_values = explainer(X_shap)
shap_values.shape


# In[ ]:


# [i for i in dir(shap_values) if not i.startswith("_")]


# In[ ]:


shap.summary_plot(shap_values)


# In[ ]:


explainer.expected_value


# In[ ]:


shap.force_plot(explainer.expected_value,shap_values= shap_values.values, features=X_shap)       


# ## RandomForest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(X_tr,y_tr)


# In[ ]:


scores = sf.evaluate_regression(rf_reg,X_te, y_te, X_train_df=X_tr, y_train=y_tr,
                            return_scores=True)


# In[ ]:


importances = sf.get_importance(rf_reg,X_test_df,top_n=50)


# ### Permutation Importance

# In[ ]:


from sklearn.inspection import permutation_importance
## Permutation importance takes a fit mode and test data. 
r = permutation_importance(linreg, X_tr, y_tr,
#                            n_repeats=30
                          )
r.keys()


# In[ ]:


## can make the mean importances into a series
permutation_importances = pd.Series(r['importances_mean'],index=X_tr.columns,
                           name = 'permutation importance')
permutation_importances


# In[ ]:


permutation_importances.sort_values().tail(20).plot(kind='barh',figsize=(6,12))


# In[ ]:


X_shap = shap.sample(X_tr,nsamples=200,random_state=321)


# In[ ]:


explainer = shap.TreeExplainer(rf_reg,X_shap)
shap_values = explainer(X_shap)


# In[ ]:


shap.summary_plot(shap_values)


# In[ ]:


shap.force_plot(explainer.expected_value,shap_values= shap_values.values, features=X_shap)       

