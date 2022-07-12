#!/usr/bin/env python
# coding: utf-8

# # Testing New Packages

# - 05/19/22 
# - Adding testing new skearn v1.1, yellowbrick, and dython

# - Yellowbrick For Teachers: 
#     - https://www.scikit-yb.org/en/latest/teaching.html 
# - Dython:
#     - http://shakedzy.xyz/dython/getting_started/examples/

# # Linear Regression with Statsmodels for Movie Revenue - Part 2

# ## Activity: Create a Linear Regression Model with Statsmodels for Revenue

# - Last Class:
#     - We started working with JUST the data data from the TMDB API for years 2000-2021. 
#     - We prepared the data for modeling
#         - Some feature engineering
#         - Our usual Preprocessing
#         - New steps for statsmodels!
#     - We fit a statsmodels linear regression.
#     
#     
# - Today:
#     - We Will inspect the model summary.
#     - We will create the visualizations to check assumptions about the residuals.
#     - We will iterate upon our model until we meet the 4 assumptions as best we can.
#     - We will discuss tactics for dealing with violations of the assumptions. 
#     - We will use our coefficients to make stakeholder recommendations (if theres time 🤞).

# > **[🕹 Click here to jump to Part 2!](#🕹-Part-2:-Checking-Model-Assumptions)**

# # 📺 Previously, on...

# ## Loading the Data

# In[1]:


import json
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
## fixing random for lesson generation
np.random.seed(321)

##import statsmodels correctly
import statsmodels.api as sm


# In[2]:


pd.set_option('display.max_columns',100)


# ### 📚 Finding & Loading Batches of Files with `glob`

# In[3]:


## Checking what data we already in our Data folder using os.listdir
import os
FOLDER = '../Data/'
file_list = sorted(os.listdir(FOLDER))
file_list


# In[4]:


df = pd.read_csv(FOLDER+'combined_tmdb_data.csv.gz',lineterminator='\n')
df


# ## Preprocessing

# In[5]:


## Columns to exclude
drop_cols = ['backdrop_path','backdrop_path','original_title','overview',
                 'poster_path','tagline','id','homepage', 'status',
                 'production_countries','video','spoken_languages',
            'original_language']
df = df.drop(columns=drop_cols)
df


# ### Feature Engineering
# 

# - Collection: convert to boolean
# - Genres: get just name and OHE
# - Cleaning Certification
# - Converting release date to year, month, and day.

# #### belongs to collection

# In[6]:


# there are 3,700+ movies that belong to collections
df['belongs_to_collection'].notna().sum()


# In[7]:


## Use .notna() to get True if it belongs to a collection
df['belongs_to_collection'] = df['belongs_to_collection'].notna()
df['belongs_to_collection'].value_counts()


# #### genre

# In[8]:


## Function to get just the genre names as a list 
import json
def get_genre_name(x):
    x = x.replace("'",'"')
    x = json.loads(x)
    
    genres = []
    for genre in x:
        genres.append(genre['name'])
    return genres


# In[9]:


## Use ourn function and exploding the new column
df['genres_list'] = df['genres'].apply(get_genre_name)
df_explode = df.explode('genres_list')
df_explode


# In[10]:


## save unique genres
unique_genres = df_explode['genres_list'].dropna().unique()

## Manually One-Hot-Encode Genres
for genre in unique_genres:
    df[f"Genre_{genre}"] = df['genres'].str.contains(genre,regex=False)    

## Drop original genre cols
df = df.drop(columns=['genres','genres_list'])
df


# #### certification

# In[11]:


## Checking Certification values
# df['certification'].value_counts(dropna=False)

# fix extra space certs
df['certification'] = df['certification'].str.strip()

## fix certification col
repl_cert = {'UR':'NR',
             'Not Rated':'NR',
             'Unrated':'NR',
             '-':'NR',
             '10':np.nan,
             'ScreamFest Horror Film Festival':'NR'}
df['certification'] = df['certification'].replace(repl_cert)
df['certification'].value_counts(dropna=False)


# #### Converting year to sep features

# In[12]:


## split release date into 3 columns
new_cols = ['year','month','day']
df[new_cols] = df['release_date'].str.split('-',expand=True)
df[new_cols] = df[new_cols].astype(float)
## drop original feature
df = df.drop(columns=['release_date'])
df


# In[13]:


df.info()


# In[14]:


# df= df.drop(columns='Genre_nan')


# ## Train Test Split

# In[15]:


drop_for_model = ['title','imdb_id','production_companies']
df = df.drop(columns=drop_for_model)
df


# In[16]:


## Make x and y variables
y = df['revenue'].copy()
X = df.drop(columns=['revenue']).copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=321)
X_train.head()


# In[17]:


X_train.isna().sum()


# In[18]:


## make cat selector and using it to save list of column names
cat_select = make_column_selector(dtype_include='object')
cat_cols = cat_select(X_train)
cat_cols


# In[19]:


## make num selector and using it to save list of column names
num_select = make_column_selector(dtype_include='number')
num_cols = num_select(X_train)
num_cols


# In[20]:


## select manually OHE cols for later
bool_select = make_column_selector(dtype_include='bool')
already_ohe_cols = bool_select(X_train)
already_ohe_cols


# In[21]:


## convert manual ohe to int
X_train[already_ohe_cols] = X_train[already_ohe_cols].astype(int)
X_test[already_ohe_cols] = X_test[already_ohe_cols].astype(int)
X_train


# In[22]:


## make pipelines
cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
                                       fill_value='MISSING'),
                         OneHotEncoder(handle_unknown='ignore', sparse=False))
num_pipe = make_pipeline(SimpleImputer(strategy='mean'),#StandardScaler()
                        )

preprocessor = make_column_transformer((num_pipe, num_cols),
                                       (cat_pipe,cat_cols),remainder='passthrough')
preprocessor


# In[23]:


## fit the col transformer
preprocessor.fit(X_train)

## Finding the categorical pipeline in our col transformer.
preprocessor.named_transformers_['pipeline-2']


# In[24]:


## B) Using list-slicing to find the encoder 
cat_features = preprocessor.named_transformers_['pipeline-2'][-1].get_feature_names_out(cat_cols)


## Create the empty list
final_features = [*num_cols,*cat_features,*already_ohe_cols]
len(final_features)


# In[25]:


preprocessor.transform(X_train).shape


# In[26]:


X_train_tf = pd.DataFrame( preprocessor.transform(X_train), 
                          columns=final_features, index=X_train.index)
X_train_tf.head()


# In[27]:


X_test_tf = pd.DataFrame( preprocessor.transform(X_test),
                         columns=final_features, index=X_test.index)
X_test_tf.head()


# # 05/19/22 Using sklearn v1.1 fixed get_feature_names_out()

# In[28]:


## make pipelines
cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
                                       fill_value='MISSING'),
                         OneHotEncoder(handle_unknown='ignore', sparse=False))
num_pipe = make_pipeline(SimpleImputer(strategy='mean'),#StandardScaler()
                        )

preprocessor = make_column_transformer((num_pipe, num_cols),
                                       (cat_pipe,cat_cols),
                                       verbose_feature_names_out=False, ## SIMPLIFIES NAMES
                                       remainder='passthrough')
preprocessor


# In[29]:


## fit the col transformer
preprocessor.fit(X_train)
X_train_tf = pd.DataFrame( preprocessor.transform(X_train), 
                          columns=preprocessor.get_feature_names_out(),
                          index=X_train.index)
X_test_tf = pd.DataFrame( preprocessor.transform(X_test), 
                          columns=preprocessor.get_feature_names_out(),
                          index=X_test.index)
X_train_tf.head()


# ## Yellowbrick

# In[30]:


# import warnings 
# warnings.filterwarnings('ignore')


# In[31]:


# import yellowbrick as yb
# from yellowbrick.features import Rank2D
# # g = yb.anscombe();
# # plt.show()


# In[32]:


# # from yellowbrick.datasets import load_credit
# from yellowbrick.features import Rank2D

# # # Load the credit dataset
# # X, y = load_credit()

# # Instantiate the visualizer with the covariance ranking algorithm
# visualizer = Rank2D(algorithm='covariance')

# visualizer.fit(X, y)           # Fit the data to the visualizer
# visualizer.transform(X)        # Transform the data
# visualizer.show()              # Finalize and render the figure


# In[33]:


X


# In[34]:


X_train_tf


# In[35]:


# visualizer = Rank2D(algorithm='covariance')
# visualizer.fit(X_train_tf,y_train)


# # PREVIOUS CONTINUED

# ## Adding a Constant for Statsmodels

# In[36]:


##import statsmodels correctly
import statsmodels.api as sm


# > Tip: make sure that add_constant actually added a new column! You may need to change the parameter `has_constant` to "add"

# In[37]:


## Make final X_train_df and X_test_df with constants added
X_train_df = sm.add_constant(X_train_tf, prepend=False, has_constant='add')
X_test_df = sm.add_constant(X_test_tf, prepend=False, has_constant='add')
display(X_train_df.head(2),X_test_df.head(2))


# # 🕹 Part 2: Checking Model Assumptions

# ## Modeling

# ### Baseline Model

# In[38]:


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
result.summary()


# In[39]:


## Get train data performance from skearn to confirm matches OLS
y_hat_train = result.predict(X_train_df)
print(f'Training R^2: {r2_score(y_train, y_hat_train):.3f}')

## Get test data performance
y_hat_test = result.predict(X_test_df)
print(f'Testing R^2: {r2_score(y_test, y_hat_test):.3f}')


# # The Assumptions of Linear Regression

# - The 4 Assumptions of a Linear Regression are:
#     - Linearity: That the input features have a linear relationship with the target.
#     - Independence of features (AKA Little-to-No Multicollinearity): That the features are not strongly related to other features.
#     - **Normality: The model's residuals are approximately normally distributed.**
#     - **Homoscedasticity: The model residuals have equal variance across all predictions.**
# 

# ### QQ-Plot for Checking for Normality

# In[40]:


result.resid


# In[41]:


## Create a Q-QPlot
resids = y_train- y_hat_train
resids


# In[42]:


## then use sm's qqplot with line='45' fit=True
sm.graphics.qqplot(resids, fit=True, line='45');


# ### Residual Plot for Checking Homoscedasticity

# In[43]:


## Plot scatterplot with y_hat_test vs resids
fig,ax =plt.subplots()
ax.scatter(y_hat_train, resids)
ax.axhline(color='k')


# ### Putting it all together into a function

# In[44]:


def evaluate_ols(result,X_train_df, y_train, show_summary=True):
    """Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.
    """
    if show_summary==True:
        try:
            display(result.summary())
        except:
            pass
    
    ## save residuals from result
    y_pred = result.predict(X_train_df)
    resid = y_train - y_pred
    
    fig, axes = plt.subplots(ncols=2,figsize=(12,5))
    
    ## Normality 
    sm.graphics.qqplot(resid,line='45',fit=True,ax=axes[0]);
    
    ## Homoscedasticity
    ax = axes[1]
    ax.scatter(y_pred, resid, edgecolor='white',lw=1)
    ax.axhline(0,zorder=0)
    ax.set(ylabel='Residuals',xlabel='Predicted Value');
    plt.tight_layout()
    


# In[45]:


evaluate_ols(result,X_train_df, y_train)


# # Improving Our Model:

# > "Garbage In = Garbage Out"
# 
# 
# - Before we dive into iterating on our model, I realized there were some big issues that I did not account for in the original data. 
#     - some movies may not have been released. 
#     - We should probably remove movies with 0 budget and revenue.
#     

# In[46]:


## reload the data
df = pd.read_csv(FOLDER+'combined_tmdb_data.csv.gz',lineterminator='\n')
df.head(2)


# ### Repeating Feature Engineering

# In[47]:


## Columns to exclude - Keeping Status and companies
drop_cols = ['backdrop_path','backdrop_path','original_title','overview',
                 'poster_path','tagline','id','homepage', #'status','production_companies'
                 'production_countries','video','spoken_languages',
            'original_language']
df = df.drop(columns=drop_cols)

## Use .notna() to get True if it belongs to a collection
df['belongs_to_collection'] = df['belongs_to_collection'].notna()

## Function to get just the genre names as a list 
import json
def get_genre_name(x):
    x = x.replace("'",'"')
    x = json.loads(x)
    
    genres = []
    for genre in x:
        genres.append(genre['name'])
    return genres

## Use ourn function and exploding the new column
df['genres_list'] = df['genres'].apply(get_genre_name)
df_explode = df.explode('genres_list')

## save unique genres
unique_genres = df_explode['genres_list'].dropna().unique()

## Manually One-Hot-Encode Genres
for genre in unique_genres:
    df[f"Genre_{genre}"] = df['genres'].str.contains(genre,regex=False)    


## Drop original genre cols
df = df.drop(columns=['genres','genres_list'])


#### Fixing Certification
## Checking Certification values
df['certification'].value_counts(dropna=False)
# fix extra space certs
df['certification'] = df['certification'].str.strip()

## fix certification col
repl_cert = {'UR':'NR',
             'Not Rated':'NR',
             'Unrated':'NR',
             '-':'NR',
             '10':np.nan,
             'ScreamFest Horror Film Festival':'NR'}
df['certification'] = df['certification'].replace(repl_cert)
df['certification'].value_counts(dropna=False)


#### Converting year to sep features
## split release date into 3 columns
new_cols = ['year','month','day']
df[new_cols] = df['release_date'].str.split('-',expand=True)
df[new_cols] = df[new_cols].astype(float)

## drop original feature
df = df.drop(columns=['release_date'])
df


# ### New Filtering

# - Make sure to only keep:
#     1. Status=Released.
#     2. Budget >0
#     3. Revenue >0

# In[48]:


## Check status
df['status'].value_counts()


# In[49]:


## Save only Released status
df = df.loc[ df['status'] == 'Released']
df = df.drop(columns=['status'])
df


# In[50]:


## filter out financials that don't have financial data
df = df.loc[(df['budget'] >0 ) & (df['revenue']>0)]
df


# ### Production Company

# In[51]:


df['production_companies']


# In[52]:


## getting longest string to check for multiple companies
idxmax = df['production_companies'].apply(len).idxmax()
idxmax 


# In[53]:


test = df.loc[idxmax, 'production_companies']
test


# In[54]:


# using regular expressions to extrap just the name
import re
exp= r"\'name\'\:.?\'(\w*.*?)\'"
re.findall(exp, test)


# In[55]:


def get_prod_company_names(x):
    if x=='[]':
        return ["MISSING"]
    
    exp= r"\'name\'\:.?\'(\w*.*?)\'"
    companies = re.findall(exp, x)
    return companies


# In[56]:


## test function
get_prod_company_names(test)


# In[57]:


## Save new clean prod_comapny col and explode
df['prod_company'] = df['production_companies'].apply(get_prod_company_names)
prod_companies = df['prod_company'].explode()
prod_companies.value_counts().head(49)


# In[58]:


prod_companies.nunique()


# In[59]:


# df['prod_company'].nunique()


# - Common Prod Company Encoding:
#     - Keep top 50 most common companies an one hot encode

# In[60]:


## saving the 50 most common companies
common_companies = sorted(prod_companies.value_counts().head(50).index)
common_companies


# In[61]:


## manually ohe top 20 companies
for company in common_companies:
    df[f"ProdComp_{company}"] = df['production_companies'].str.contains(company, regex=False)


# In[62]:


## Dropping columns
drop_for_model = ['title','imdb_id','prod_company','production_companies']
df = df.drop(columns=drop_for_model)
df


# # Checking for Linearity

# In[63]:


## concatenating training data into plot_df
plot_df = pd.concat([X_train_df,y_train],axis=1)
plot_df


# In[64]:


## save plot_cols list to show (dropping genre from plot_df from pair_plot)
genre_cols = [c for c in plot_df if c.startswith('Genre')]
comp_cols = [c for c in plot_df if c.startswith('ProdComp')]

plot_cols = plot_df.drop(columns=[*genre_cols, *comp_cols]).columns
plot_cols


# In[65]:


len(plot_cols)


# In[66]:


## Plot first 6 features
sns.pairplot(data=plot_df, y_vars='revenue', x_vars=plot_cols[:6])


# In[67]:


## Plot next 6 features
sns.pairplot(data=plot_df, y_vars='revenue', x_vars=plot_cols[6:13])


# In[68]:


## plot remaining features
sns.pairplot(data=plot_df, y_vars='revenue', x_vars=plot_cols[13:])


# - Shouldn't have years before 2000, so drop. 
# - Check outliers in popularity, runtime
# 

# In[69]:


# remove movies prior to 2000
df = df.loc[ df['year']>=2000]
df


# In[70]:


## plot remaining features
sns.pairplot(data=plot_df, y_vars='revenue', x_vars=plot_cols[13:])


# > Now need to recreate X and y varaibles

# ### Functionize ALL of the preprocessing

# In[71]:


def get_train_test_split(df_, y_col='revenue',drop_cols=[]):
    
    ## Make copy of input df
    df = df_.copy()
    
    ## filter columns in drop cols (if exist)
    final_drop_cols = []
    [df.drop(columns=c,inplace=True) for c in df.columns if c in drop_cols]
    
    
    ## Make x and y variables
    y = df[y_col].copy()
    X = df.drop(columns=[y_col]).copy()

    X_train, X_test, y_train, y_test = train_test_split(X,y)#, random_state=321)
    

    
    ## make cat selector and using it to save list of column names
    cat_select = make_column_selector(dtype_include='object')
    cat_cols = cat_select(X_train)


    ## make num selector and using it to save list of column names
    num_select = make_column_selector(dtype_include='number')
    num_cols = num_select(X_train)


    ## select manually OHE cols for later
    bool_select = make_column_selector(dtype_include='bool')
    already_ohe_cols = bool_select(X_train)

    ## convert manual ohe to int
    X_train[already_ohe_cols] = X_train[already_ohe_cols].astype(int)
    X_test[already_ohe_cols] = X_test[already_ohe_cols].astype(int)

    ## make pipelines
    cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
                                           fill_value='MISSING'),
                             OneHotEncoder(handle_unknown='ignore', sparse=False))
    num_pipe = make_pipeline(SimpleImputer(strategy='mean'),#StandardScaler()
                            )

    preprocessor = make_column_transformer((num_pipe, num_cols),
                                           (cat_pipe,cat_cols),remainder='passthrough')
    
    

    ## fit the col transformer
    preprocessor.fit(X_train)

    ## Finding the categorical pipeline in our col transformer.
    preprocessor.named_transformers_['pipeline-2']

    ## B) Using list-slicing to find the encoder 
    cat_features = preprocessor.named_transformers_['pipeline-2'][-1].get_feature_names_out(cat_cols)


    ## Create the empty list
    final_features = [*cat_features,*num_cols,*already_ohe_cols]

    ## Make df verisons of x data
    X_train_tf = pd.DataFrame( preprocessor.transform(X_train), 
                              columns=final_features, index=X_train.index)


    X_test_tf = pd.DataFrame( preprocessor.transform(X_test),
                             columns=final_features, index=X_test.index)


    ### Adding a Constant for Statsmodels
    ## Make final X_train_df and X_test_df with constants added
    X_train_df = sm.add_constant(X_train_tf, prepend=False, has_constant='add')
    X_test_df = sm.add_constant(X_test_tf, prepend=False, has_constant='add')
    return X_train_df, y_train, X_test_df, y_test


# ### Model #1

# In[72]:


## Use our function to make new x,y vars
X_train_df, y_train, X_test_df, y_test = get_train_test_split(df)
X_train_df

## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
evaluate_ols(result, X_train_df, y_train)


# > How did we do? Did we meet the assumptions better?

# ## Removing Outliers

# ### Using Z-Score Rule

# In[73]:


from scipy import stats
def find_outliers(data, verbose=True):
    outliers = np.abs(stats.zscore(data))>3
    
    if verbose:
        print(f"- {outliers.sum()} outliers found in {data.name} using Z-Scores.")
    return outliers


# In[74]:


find_outliers(df['runtime'])


# In[75]:


## save a dictionary of the T/F outlier index for each feature in outleir_cols
outlier_cols = ['runtime','popularity','revenue']

outliers = {}

for col in outlier_cols:
    col_outliers = find_outliers(df[col])
    
    outliers[col] = col_outliers


# In[76]:


# Make new df_clean copy of df
df_clean = df.copy()

## loop through dictionary to remove outliers
for col, col_outliers in outliers.items():
    df_clean  = df_clean.loc[~col_outliers]


# In[77]:


df_clean


# ### Model 2: Outliers Removed (Z_scores)

# In[78]:


## Use our function to make new x,y vars
X_train_df, y_train, X_test_df, y_test = get_train_test_split(df_clean)


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
evaluate_ols(result, X_train_df, y_train)


# ### Removing Outliers - Using IQR Rule

# In[79]:


## visualzie outlier-removed target
sns.boxplot(x = y_train)


# In[80]:


def find_outliers_IQR(data, verbose=True):
    q3 = np.quantile(data,.75)
    q1 = np.quantile(data,.25)

    IQR = q3 - q1
    upper_threshold = q3 + 1.5*IQR
    lower_threshold = q1 - 1.5*IQR
    
    outliers = (data<lower_threshold) | (data>upper_threshold)
    if verbose:
        print(f"- {outliers.sum()} outliers found in {data.name} using IQR.")
        
    return outliers


# In[81]:


outliers_z = find_outliers(df['revenue'])
outliers_iqr = find_outliers_IQR(df['revenue'])


# In[82]:


## Loop to remove outliers from same clumns using new function
outlier_cols = ['runtime','popularity','revenue']

## Empty dict for both types of outliers
## save a dictionary of the T/F outlier index for each feature in outleir_cols
outlier_cols = ['runtime','popularity','revenue']

outliers_z = {}
outliers_IQR = {}

for col in outlier_cols:
    col_outliers_z = find_outliers(df[col])
    outliers_z[col] = col_outliers_z
    
    col_outliers_iqr = find_outliers_IQR(df[col])
    outliers_IQR[col] = col_outliers_iqr
    print()
## Use both functions to see the comparison for # of outliers


# In[83]:


# # remove_outliers - create df_clean_z
# df_clean_z = df.copy()

# ## loop though outliers_z


# In[84]:


# Make new df_clean copy of df
df_clean_iqr = df.copy()

## loop through dictionary to remove outliers
for col, col_outliers in outliers_IQR.items():
    df_clean_iqr  = df_clean_iqr.loc[~col_outliers]
df_clean_iqr


# ### Model 3 - IQR Outliers Removed

# In[85]:


## Use our function to make new x,y vars
X_train_df, y_train, X_test_df, y_test = get_train_test_split(df_clean_iqr)
X_train_df

## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
evaluate_ols(result, X_train_df, y_train)


# > How are we doing??

# In[86]:


## get pvalues from model result
pvalues =result.pvalues
pvalues


# In[87]:


## Check for p-values that are >.05
pvalues[pvalues>.05]


# ## Removing features - based on p-values

# In[88]:


## Get list of ALL genre columns to see how many are sig
genre_cols


# In[89]:


## save just genre pvalues
genre_pvals = pvalues[genre_cols]
genre_pvals


# In[90]:


(genre_pvals>.05).sum() /len(genre_pvals)


# In[91]:


## calc what % are insig?
len(genre_pvals>.05)/len(genre_pvals)


# In[ ]:





# In[92]:


comp_cols = [c for c in df_clean_iqr if c.startswith('ProdComp')]


# In[93]:


comp_cols


# In[94]:


## Get list of ALL prod_comp columns to see how many are sig
comp_pvals = pvalues[comp_cols]
(comp_pvals>.05).sum()


# In[95]:


len(comp_pvals>.05)/len(comp_pvals)


# In[96]:


## save just genre pvalues


# > both have <50% bad pvalues. Keep both!

# In[97]:


## what pvals are remaining?


# ### Model 4

# In[98]:


df_clean_iqr = df_clean_iqr.drop(columns=[*genre_cols, *comp_cols])
df_clean_iqr


# In[99]:


## Use our function to make new x,y vars
X_train_df, y_train, X_test_df, y_test = get_train_test_split(df_clean_iqr)

## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
evaluate_ols(result, X_train_df, y_train)


# # Addressing Multicollinearity

# In[100]:


## Calculating the mask to hide the upper-right of the triangle
corr = X_train_df.corr().abs()

mask = np.triu(np.ones_like(corr))

plt.figure(figsize=(25,25))
sns.heatmap(corr,square=True, cmap='Reds', annot=True, mask=mask);


# ### Variance Inflation Factor

# In[101]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
 
# separate just x-data and subtract mean
features = X_train_df -  X_train_df.mean()

features


# In[102]:


# create a list of VIF scores for each feature in features.
vif_scores = [variance_inflation_factor(features.values, i) for i in range(len(features.columns))]

# create a new dataframe to hold the VIF scores 
VIF = pd.Series(vif_scores, index=features.columns)
VIF


# In[103]:


## Visualize Coefficients
ax = result.params.sort_values().plot(kind='barh')
ax.axvline()


# ## Compare to Alternative Regressors

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
X_train_df, y_train, X_test_df, y_test = get_train_test_split(df)

reg = RandomForestRegressor(verbose=1,random_state=42)
reg.fit(X_train_df, y_train)


# In[ ]:


## Get train data performance from skearn to confirm matches OLS
y_hat_train = reg.predict(X_train_df)
print(f'Training R^2: {r2_score(y_train, y_hat_train):.3f}')

## Get test data performance
y_hat_test = reg.predict(X_test_df)
print(f'Testing R^2: {r2_score(y_test, y_hat_test):.3f}')


# In[ ]:


evaluate_ols(reg, X_train_df, y_train)


# In[ ]:


importances = pd.Series(reg.feature_importances_, index=X_train_df.columns)
importances.sort_values().tail(25).plot(kind='barh',figsize=(6,10))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




