#!/usr/bin/env python
# coding: utf-8

# # Comparing Datasets - Linear Regression

# ## Datasets to Compare:

# - King's County
# - AMES
# - Boston
# 
# 

# # Code

# ## Loading in datasets csv 
# - from my ["Important coding dojo data sciecnce Link"](https://docs.google.com/spreadsheets/d/1BIR_4P2fJEJPIM_mWgyTdOEy4mOFM7YLN2MtdjlDvSI/edit?usp=sharing) Google Sheet

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


# In[2]:


df_data = pd.read_csv('Data/James Dataset links.csv')
df_data


# In[3]:


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
## Customized Options
pd.set_option('display.float_format',lambda x: f"{x:,.4f}")
plt.style.use('seaborn-talk')


# In[4]:


##import statsmodels correctly
import statsmodels.api as sm
from scipy import stats


# In[5]:


to_find = ['ames','king county','boston']

URLS = {}

for name in to_find:
    is_dataset = df_data['Name'].str.contains(name, case=False)
    URLS[name] = df_data.loc[is_dataset,'FULL URL'].tolist()[0
                                                            ]

URLS


# In[6]:


from pandas_profiling import ProfileReport


# ### Functions

# > Moved to stack_functions.py

# In[7]:


## Adding folder above to path
import os, sys
sys.path.append(os.path.abspath('../'))

## Load stack_functions with autoreload turned on
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import stack_functions as sf

def show_code(function):
    import inspect 
    from IPython.display import display,Markdown, display_markdown
    code = inspect.getsource(function)
    md_txt = f"```python\n{code}\n```"
    return display(Markdown(md_txt))
    


# In[8]:


show_code(sf.find_outliers_IQR)
show_code(sf.find_outliers_Z)
show_code(sf.remove_outliers)


# In[ ]:


show_code(sf.evaluate_ols)
show_code(sf.plot_coeffs)
show_code(sf.get_importance)


# # King's County

# In[ ]:


df = pd.read_csv(URLS['king county'])
df = df.drop(columns=['id','date'])
## Dropping some some features
df.info()
df.head()


# In[ ]:


# report_king = ProfileReport(df)
# report_king


# In[ ]:


## Treating zipcode as a category
df['zipcode'] = df['zipcode'].astype(str)


# ## Full Dataset Preprocessing

# In[ ]:


## Make x and y variables
target = 'price'
drop_cols = ['view']


y = df[target].copy()
X = df.drop(columns=[target,*drop_cols]).copy()

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


## Save list of zipcode columns and other columns
zip_cols = [c for c in X_train_df.columns if c.startswith('zipcode')]
nonzip_cols = X_train_df.drop(columns=[*zip_cols]).columns.tolist()

X_test_df.head()


# In[ ]:


X_train_df.describe()


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
zip_cols = [c for c in X_train_df.columns if c.startswith('zipcode')]
nonzip_cols = X_train_df.drop(columns=[*zip_cols]).columns.tolist()


X_test_scaled_df.head()


# In[ ]:


X_train_scaled_df.describe()


# ## Modeling

# ### Raw Numeric - No Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
sf.evaluate_ols(result,X_train_df, y_train)


# In[ ]:


fig_raw =sf.plot_coeffs(result, zip_cols, include_const=True,title="Raw Coefficients")


# In[ ]:


fig_raw_zips =sf.plot_coeffs(result, nonzip_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# ### Raw Numeric - with Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df_cnst)

## Fit the model and view the summary
result = model.fit()
sf.evaluate_ols(result,X_train_df_cnst, y_train)


# In[ ]:


fig_raw =sf.plot_coeffs(result, zip_cols, include_const=True,title="Raw Coefficients")


# In[ ]:


fig_raw =sf.plot_coeffs(result, zip_cols, include_const=False,title="Raw Coefficients")


# In[ ]:


fig_raw_zips =sf.plot_coeffs(result, nonzip_cols, figsize=(6,12),include_const=False,title="Raw Coefficients")


# In[ ]:





# ### Scaled Numeric - No Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_scaled_df)

## Fit the model and view the summary
result_scaled = model.fit()
sf.evaluate_ols(result_scaled,X_train_scaled_df, y_train)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_scaled,zip_cols,include_const=True)


# In[ ]:


fig_scaled_zips =sf.plot_coeffs(result_scaled, nonzip_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# ### Scaled Numeric - with Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_scaled_df_cnst)

## Fit the model and view the summary
result_scaled = model.fit()
sf.evaluate_ols(result_scaled,X_train_scaled_df_cnst, y_train)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_scaled,zip_cols,include_const=True)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_scaled,zip_cols,include_const=True)


# In[ ]:


fig_scaled_zips =sf.plot_coeffs(result_scaled, nonzip_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# ### Observations

# - The bedrooms coefficient being negative is problematic
#     - [ ] Try not including a constant/intercept.

# In[ ]:


sns.barplot(data=df, x='bedrooms',y='price')


# In[ ]:


sns.regplot(data=df, x='bedrooms',y='price',scatter_kws={'ec':'w','lw':1})


# In[ ]:


ax = sns.barplot(data=df, x='bedrooms',y='price',edgecolor='k',linewidth=2)
sns.stripplot(data=df, x='bedrooms',y='price',edgecolor='gray',linewidth=0.5,
              alpha=0.5,)


# In[ ]:


sns.countplot(data=df, x='bedrooms')#,y='price')


# In[ ]:


sns.boxplot(data=df, x='bedrooms',y='price')


# ## Removing Outliers

# In[ ]:


df_clean = sf.remove_outliers(df, method='z')
df_clean


# In[ ]:


## Make x and y variables
target = 'price'
drop_cols = ['view']

y_cln = df_clean[target].copy()
X_cln = df_clean.drop(columns=[target,*drop_cols]).copy()

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
zip_cols = [c for c in X_train_df_cln.columns if c.startswith('zipcode')]
nonzip_cols = X_train_df_cln.drop(columns=[*zip_cols]).columns.tolist()

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
zip_cols = [c for c in X_train_scaled_df_cln.columns if c.startswith('zipcode')]
nonzip_cols = X_train_scaled_df_cln.drop(columns=[*zip_cols]).columns.tolist()


X_test_scaled_df.head()


# In[ ]:


X_train_scaled_df.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# ## Make x and y variables
# target = 'price'
# drop_cols = ['view']


# y_cln = df_clean[target].copy()
# X_cln = df_clean.drop(columns=[target,*drop_cols]).copy()

# X_train_cln, X_test_cln, y_train_cln, y_test_cln = train_test_split(X_cln,y_cln, random_state=321)
# X_train_cln.head()


# In[ ]:


# ## cat selector
# cat_sel = make_column_selector(dtype_include='object')
# cat_cols = cat_sel(X_cln)

# # num selectorr
# num_sel = make_column_selector(dtype_include='number')
# num_cols = num_sel(X_cln)

# ## make pipelines & column transformer - raw numeric
# cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
#                                        fill_value='MISSING'),
#                          OneHotEncoder(handle_unknown='ignore', sparse=False))
# num_pipe_raw = make_pipeline(SimpleImputer(strategy='mean'))
# preprocessor = make_column_transformer((num_pipe_raw, num_sel),
#                                        (cat_pipe,cat_sel), verbose_feature_names_out=False)
# preprocessor


# In[ ]:


# ## Prepare X_train_df
# X_train_df_cln = pd.DataFrame( preprocessor.fit_transform(X_train_cln), 
#                           columns=preprocessor.get_feature_names_out(),
#                          index=X_train.index)
# # X_train_df = sm.add_constant(X_train_df, prepend=False, has_constant='add')

# ## Prepare X_test_df
# X_test_df_cln = pd.DataFrame( preprocessor.transform(X_test_cln),
#                          columns=preprocessor.get_feature_names_out(), 
#                          index=X_test.index)
# # X_test_df = sm.add_constant(X_test_df, prepend=False, has_constant='add')


# X_test_df_cln.head()


# In[ ]:


# X_train_df_cln.describe()


# ## Modeling - No Outliers

# ### Raw Numeric - No Constant

# In[ ]:


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train_cln, X_train_df_cln)

## Fit the model and view the summary
result = model.fit()
sf.evaluate_ols(result,X_train_df_cln, y_train_cln)


# In[ ]:


fig_raw =sf.plot_coeffs(result, zip_cols, include_const=True,title="Raw Coefficients")


# In[ ]:


fig_raw_zips =sf.plot_coeffs(result, nonzip_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# ### Raw Numeric - with Constant

# In[ ]:


X_train_df_cln_cnst.describe()


# In[ ]:


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train_cln, X_train_df_cln_cnst)

## Fit the model and view the summary
result = model.fit()
sf.evaluate_ols(result,X_train_df_cln_cnst, y_train_cln)


# In[ ]:


fig_raw =sf.plot_coeffs(result, zip_cols, include_const=True,title="Raw Coefficients")


# In[ ]:


fig_raw =sf.plot_coeffs(result, zip_cols, include_const=False,title="Raw Coefficients")


# In[ ]:


fig_raw_zips =sf.plot_coeffs(result, nonzip_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


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
model = sm.OLS(y_train_cln, X_train_scaled_df_cln)

## Fit the model and view the summary
result_scaled = model.fit()
sf.evaluate_ols(result_scaled,X_train_scaled_df_cln, y_train_cln)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_scaled,zip_cols,include_const=True)


# In[ ]:


fig_scaled_zips =sf.plot_coeffs(result_scaled, nonzip_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# In[ ]:


# sns.regplot(data=df_clean, x='bedrooms',y='price',scatter_kws={'ec':'w','lw':1})


# In[ ]:


# ax = sns.barplot(data=df_clean, x='bedrooms',y='price',edgecolor='k',linewidth=2)
# sns.stripplot(data=df_clean, x='bedrooms',y='price',edgecolor='gray',linewidth=0.5,
#               alpha=0.5,)


# ### Scaled Numeric - with Constant

# In[ ]:


X_train_scaled_df_cln_cnst.describe()


# In[ ]:


## instantiate an OLS model WITH the training data.
model_scaled = sm.OLS(y_train_cln, X_train_scaled_df_cln_cnst)

## Fit the model and view the summary
result_scaled = model_scaled.fit()
sf.evaluate_ols(result_scaled,X_train_scaled_df_cln_cnst, y_train_cln)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_scaled,zip_cols,include_const=True)


# In[ ]:


fig_scaled =sf.plot_coeffs(result_scaled,zip_cols,include_const=False)


# ## Feature Importance - Random Forest

# In[ ]:


# ## Make x and y variables
# target = 'price'
# drop_cols = ['view']


# y = df[target].copy()
# X = df.drop(columns=[target,*drop_cols]).copy()

# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=321)
# X_train.head()


# ### Raw Numeric - No Constant

# In[ ]:


# ## cat selector
# cat_sel = make_column_selector(dtype_include='object')
# cat_cols = cat_sel(X)

# # num selectorr
# num_sel = make_column_selector(dtype_include='number')
# num_cols = num_sel(X)

# ## make pipelines & column transformer - raw numeric
# cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
#                                        fill_value='MISSING'),
#                          OneHotEncoder(handle_unknown='ignore', sparse=False))
# num_pipe_raw = make_pipeline(SimpleImputer(strategy='mean'))
# preprocessor = make_column_transformer((num_pipe_raw, num_sel),
#                                        (cat_pipe,cat_sel), verbose_feature_names_out=False)
# preprocessor


# In[ ]:


# ## Prepare X_train_df
# X_train_df = pd.DataFrame( preprocessor.fit_transform(X_train), 
#                           columns=preprocessor.get_feature_names_out(),
#                          index=X_train.index)
# # X_train_df = sm.add_constant(X_train_df, prepend=False, has_constant='add')

# ## Prepare X_test_df
# X_test_df = pd.DataFrame( preprocessor.transform(X_test),
#                          columns=preprocessor.get_feature_names_out(), 
#                          index=X_test.index)
# # X_test_df = sm.add_constant(X_test_df, prepend=False, has_constant='add')


# X_test_df.head()


# In[ ]:


## instantiate an OLS model WITH the training data.
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train_df,y_train)

## Fit the model and view the summary
sf.evaluate_ols(rf_reg,X_train_df, y_train)


# ## Permutation Importance

# In[ ]:


## instantiate an OLS model WITH the training data.
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train_scaled_df,y_train)

## Fit the model and view the summary
sf.evaluate_ols(linreg,X_train_scaled_df, y_train)


# In[ ]:


from sklearn.inspection import permutation_importance
y_test


# In[ ]:


## Permutation importance takes a fit mode and test data. 
r = permutation_importance(linreg, X_test_scaled_df, y_test,
                           n_repeats=30)
r.keys()


# In[ ]:


r['importances_mean']


# In[ ]:


## can make the mean importances into a series
permutation_importances = pd.Series(r['importances_mean'],index=X_train_df.columns,
                           name = 'permutation importance')
permutation_importances


# In[ ]:


permutation_importances.drop(zip_cols).sort_values().plot(kind='barh')


# In[ ]:


fig_scaled_zips =sf.plot_coeffs(result_scaled, nonzip_cols, figsize=(6,12),include_const=True,title="Raw Coefficients")


# # Testing Assignment Data

# ## Discussed with Brenda (Kaggle)
# - [One on One Doc](https://docs.google.com/document/d/1yCqejdZHTByU-gIfGtEcnH_P4cSvJh7Az_8EtyjeGBI/edit?usp=sharing)
#     - See 06/21/22

# In[ ]:


URLS['students'] = 'Data/student-mat.csv' # use for converting regr to classificaton
URLS['insurance'] = "Data/insurance.csv"
URLS


# In[ ]:


pd.read_csv(URLS['students'])


# In[ ]:


# pd.read_csv(URLS['insurance'])


# In[ ]:


# URLS['student-performance']


# ## Student Performance

# - Predict...."G3" using other features.
#     - Could also combine G1-G3 as mean
#     - for class I calculated the % scores instead of the out of 20 scores

# In[ ]:


df = pd.read_csv(URLS['students'])
df.info()
df


# In[ ]:


df.nunique()


# In[ ]:


ProfileReport(df)


# In[ ]:


## Make x and y variables
target = 'G3'
drop_cols = []#'view']


y = df[target].copy()
X = df.drop(columns=[target,*drop_cols]).copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=321)
X_train.head()


# ### Raw Numeric - No Constant

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


## Prepare X_train_df
X_train_df = pd.DataFrame( preprocessor.fit_transform(X_train), 
                          columns=preprocessor.get_feature_names_out(),
                         index=X_train.index)
# X_train_df = sm.add_constant(X_train_df, prepend=False, has_constant='add')

## Prepare X_test_df
X_test_df = pd.DataFrame( preprocessor.transform(X_test),
                         columns=preprocessor.get_feature_names_out(), 
                         index=X_test.index)
# X_test_df = sm.add_constant(X_test_df, prepend=False, has_constant='add')


X_test_df.head()


# In[ ]:


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
sf.evaluate_ols(result,X_train_df, y_train)


# In[ ]:


sf.plot_coeffs(result,include_const=True,figsize=(6,12));


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


X_train_scaled_df = pd.DataFrame( preprocessor_scale.fit_transform(X_train), 
                          columns=preprocessor_scale.get_feature_names_out(),
                         index=X_train.index)
X_train_scaled_df = sm.add_constant(X_train_scaled_df, prepend=False, has_constant='add')


X_test_scaled_df = pd.DataFrame( preprocessor_scale.transform(X_test),
                         columns=preprocessor_scale.get_feature_names_out(), 
                         index=X_test.index)
X_test_scaled_df = sm.add_constant(X_test_scaled_df, prepend=False, has_constant='add')

X_test_scaled_df.head()


# In[ ]:


## instantiate an OLS model WITH the training data.
model_scaled = sm.OLS(y_train, X_train_scaled_df)

## Fit the model and view the summary
result_scaled = model_scaled.fit()
sf.evaluate_ols(result_scaled,X_train_scaled_df, y_train)


# In[ ]:


sf.plot_coeffs(result_scaled,include_const=True,figsize=(6,12));


# In[ ]:





# # OLD

# In[ ]:


raise Exception("STOP HERE!")


# # Multicollinearity 

# In[ ]:


zip_cols = [c for c in X_train_df.columns if c.startswith('zipcode')]


# In[ ]:


## Calculating the mask to hide the upper-right of the triangle
plt.figure(figsize=(15,15))
corr = X_train_df.drop(columns=zip_cols).corr().abs()
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr,square=True, cmap='Reds', annot=True, mask=mask);


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
 
# separate just x-data and subtract mean
features = X_train_df -  X_train_df.mean()

features


# In[ ]:


# create a list of VIF scores for each feature in features.
vif_scores = [variance_inflation_factor(features.values, i) for i in range(len(features.columns))]

# create a new dataframe to hold the VIF scores 
VIF = pd.Series(vif_scores, index=features.columns)
VIF


# In[ ]:


## Set float format to view vals not in scientfic notation
pd.set_option('display.float_format',lambda x: f'{x:.2f}')


# In[ ]:


## remove infinite values from VIF and sort
VIF = VIF[VIF!=np.inf].sort_values()
VIF


# In[ ]:


## filter for VIF that are > 5
VIF[VIF>5]


# In[ ]:


high_vif = VIF[VIF>5].index
high_vif


# In[ ]:


## Get train data performance from skearn to confirm matches OLS
y_hat_train = result.predict(X_train_df)
print(f'Training R^2: {r2_score(y_train, y_hat_train):.3f}')

## Get test data performance
y_hat_test = result.predict(X_test_df)
print(f'Testing R^2: {r2_score(y_test, y_hat_test):.3f}')


# In[ ]:




