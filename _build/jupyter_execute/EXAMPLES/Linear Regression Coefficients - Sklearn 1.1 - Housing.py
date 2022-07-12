#!/usr/bin/env python
# coding: utf-8

# # Intro to Linear to Logistic Regression Coefficients

# - 06/15/22
# - James Irving

# ## Learning Objectives
# 

# 
# - To review how linear regression predicts a continuous value.
# - To understand what coefficients are and how they are used to calcualte the target.

# - Lesson Duration:
#     - ~10 mins

# # Predicting the Price of a Home Using Linear Regression

# <img src="https://github.com/jirvingphd/from-linear-to-logistic-regression-brief-intro/blob/main/images/istock24011682medium_1200xx1697-955-0-88.jpg?raw=1" width=60% alt="Source: https://images.app.goo.gl/oJoMSGU8LGgDjkA76">

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
## Customization Options
pd.set_option('display.float_format',lambda x: f"{x:,.4f}")
plt.style.use('seaborn-talk')
plt.rcParams['figure.facecolor']='white'


# In[2]:


## additional required imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import metrics

## fixing random for lesson generation
np.random.seed(321)

##import statsmodels correctly
import statsmodels.api as sm
## Customized Options
pd.set_option('display.float_format',lambda x: f"{x:,.4f}")
plt.style.use('seaborn-talk')


# In[3]:


## Load in the King's County housing dataset and display the head and info
df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSEZQEzxja7Hmj5tr5nc52QqBvFQdCAGb52e1FRK1PDT2_TQrS6rY_TR9tjZjKaMbCy1m5217sVmI5q/pub?output=csv")
display(df.head(),df.info())


# In[4]:


# ## FILTERING EXTREME VALUES FOR DEMONSTRATION PURPOSES
# df = df.loc[( df['bedrooms'] <8) & (df['price'] < 2_000_000) & df['bathrooms']>0]
# df


# In[5]:


## Visualize the distribution of house prices (using seaborn!)
sns.displot(df['price'],aspect=2);


# ## Visualizing Our Features vs Our Target

# - We want to determine how features of a home influence its sale price. 
# - Specifically, we will be using:
#     - `sqft_living`:Square-Footage of all Living Areas
#     - `bedrooms`: # of Bedrooms
#     - `bathrooms`: # of Bathrooms

# In[6]:


from matplotlib.ticker import StrMethodFormatter
## Plot a scatter plot of sqft-living vs price
ax = sns.scatterplot(data=df,x='sqft_living',y='price',s=50)
ax.set_title('Relationship Between Square Footage and House Price')

## Formatting Price Axis
price_fmt = StrMethodFormatter("${x:,.0f}")
ax.yaxis.set_major_formatter(price_fmt)
ax.get_figure().set_size_inches(10,6)


# - We can see a positive relationship between sqft-living and price, but it would be better if we could show the line-of-best-fit with it

# ### Functionizing Our Code

# In[7]:


## NOTE: if we had more time, we would write this together.
def plot_feature_vs_target(df,x='sqft_living',y='price',price_format=True):
    """Plots a seaborn regplot of x vs y."""
    ax = sns.regplot(data=df,x=x,y=y,
                line_kws=dict(color='k',ls='--',lw=2),
               scatter_kws=dict(s=50,edgecolor='white',lw=1,alpha=0.8)
                    )
    
    ax.get_figure().set_size_inches(10,6)
    ax.set_title(f'{x} vs {y}')
    ax.get_figure().set_facecolor('white')
    
    if price_format:
        ## Formatting Price Axis
        price_fmt = StrMethodFormatter("${x:,.0f}")
        ax.yaxis.set_major_formatter(price_fmt)
    return ax


# In[8]:


## Visualize the relationship between sqft_living and price
ax = plot_feature_vs_target(df,x='sqft_living');


# ### What Our Trendline Tells Us
# - Our trendline summarizes the relationship between our feature and our target.
# - It is comprised of the: <br>
# 1) y-intercept (AKA $c$ or $b$ or $\beta_{0}$) indicating the default value of y when X=0.<br>
# 2) and a slope (AKA $m$ or $\beta$) indicating the relationship between X and y. When X increases by 1, y increases by $m$.

# In[9]:


## Visualize the relationship between bathrooms and price
plot_feature_vs_target(df,x='bathrooms');


# In[10]:


## Visualize the relationship between bedrooms and price
plot_feature_vs_target(df,x='bedrooms')


# >- Now, let's create a Linear Regression model with sci-kit learn to determine the effect of these 3 features!

# ## Predicting House Price with sci-kit learn's `LinearRegression`

# In[11]:


# ## Create our X & y using bedrooms,bathrooms, sqft-living
# use_cols = ['bedrooms','bathrooms','sqft_living']
# X = df[use_cols].copy()
# y = df['price'].copy()

# ## Train test split (random-state 321, test_size=0.25)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=321)
# X_train


# In[12]:


# ## import LinearRegression from sklearn and fit the model
# from sklearn.linear_model import LinearRegression
# linreg = LinearRegression()
# linreg.fit(X_train,y_train)


# In[13]:


# ## Get our models' R-squared value for the train and test data
# print(f"Training R-Squared: {linreg.score(X_train,y_train):.3f}")
# print(f"Test R-Squared: {linreg.score(X_test,y_test):.3f}")


# >- Ok, so what does this tell us?
#     - Our model can explain 52% of the variance of house price using just 3 features!

# ### What Coefficients Did Our Model Find? 

# In[14]:


# linreg.coef_


# In[15]:


# linreg.intercept_


# In[16]:


# ## NOTE: with more time, we would code this together. 
# def get_coeffs(reg,X_train):
#     """Extracts the coefficients from a scikit-learn LinearRegression or LogisticRegression"""
#     coeffs = pd.Series(reg.coef_.flatten(),index=X_train.columns)

#     coeffs.loc['intercept'] = reg.intercept_

#     return coeffs


# - Linear Regression Equation
# $$ \large \hat y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n  $$
# which we can simplify to:
# $$ \hat y =  \sum_{i=0}^{N} \beta_i x_i  $$

# In[17]:


# ## Get the coefficents from the model using our new function
# coeffs = get_coeffs(linreg,X_train)
# coeffs


# >- **Each coefficient tells us the effect of increasing the values in that column by 1 unit.** 
# >- According to our model, we can determine a home's price using the following results:
#     - The model assumed a default/starting house price was \$130,191.2155 (the intercept)
#     - For each additional bedrooms, subtract      \$-41,206.78
#     - For each batrhoom, add \$13,537.01
#     - For each square foot of living space, add \$243.11

# In[18]:


# ## Let's select an example house and see how we calculate price
# i = 300
# house = X_test.iloc[i]
# house


# In[19]:


# ## Calculate the home's predicted price using our coefficients
# price = house['bedrooms']*coeffs['bedrooms'] + \
#         house['bathrooms']*coeffs['bathrooms'] + \
#         house['sqft_living']*coeffs['sqft_living'] + coeffs['intercept']

# print(f"${price:,.2f}")


# In[20]:


# coeffs.values


# In[21]:


# ## What would our model predict for our test house?
# linreg.predict(house.values.reshape(1,-1))


# In[22]:


# y_test.iloc[i]


# ## Linear Regression Summary
# - Linear regression allowed us to predict the exact dollar price of a given home.
# - It summarizes the relationship of each feature using coefficients, which are used to calculate the target. 
# 
# >-  But what do we do when we want to predict what group a house belongs to instead of an exact price?

# # 📚 Comparing Coefficients & Using Scikit-Learn v1.1
# 

# - 06/21/22

# In[23]:


from scipy import stats

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[24]:


## Load in the King's County housing dataset and display the head and info
df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSEZQEzxja7Hmj5tr5nc52QqBvFQdCAGb52e1FRK1PDT2_TQrS6rY_TR9tjZjKaMbCy1m5217sVmI5q/pub?output=csv",
                #  index_col=0
                 )
display(df.head(),df.info())


# In[25]:


## Dropping some features for time
df = df.drop(columns=['date','view','id'])
df


# In[26]:


## Treating zipcode as a category
df['zipcode'] = df['zipcode'].astype(str)


# ### Train Test Split

# In[27]:


## Make x and y variables
y = df['price'].copy()
X = df.drop(columns=['price']).copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=321)
X_train.head()


# In[28]:


## cat selector
cat_sel = make_column_selector(dtype_include='object')
cat_cols = cat_sel(X)
cat_cols


# In[29]:


# num selectorr
num_sel = make_column_selector(dtype_include='number')
num_cols = num_sel(X)
num_cols


# In[30]:


## make pipelines
cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
                                       fill_value='MISSING'),
                         OneHotEncoder(handle_unknown='ignore', sparse=False))
num_pipe = make_pipeline(SimpleImputer(strategy='mean'),#StandardScaler()
                        )

preprocessor = make_column_transformer((num_pipe, num_cols),
                                       (cat_pipe,cat_cols),remainder='passthrough')
preprocessor


# In[31]:


X_train.isna().sum()


# In[32]:


## fit the col transformer
preprocessor.fit(X_train)

## Finding the categorical pipeline in our col transformer.
preprocessor.named_transformers_['pipeline-2']


# In[33]:


## B) Using list-slicing to find the encoder 
cat_features = preprocessor.named_transformers_['pipeline-2'][-1].get_feature_names_out(cat_cols)


## Create the empty list
final_features = [*num_cols,*cat_features]
len(final_features)


# In[34]:


X_train_tf = pd.DataFrame( preprocessor.transform(X_train), 
                          columns=final_features, index=X_train.index)
X_train_tf.head()


# In[35]:


X_test_tf = pd.DataFrame( preprocessor.transform(X_test),
                         columns=final_features, index=X_test.index)
X_test_tf.head()


# In[36]:


##import statsmodels correctly
import statsmodels.api as sm


# In[37]:


## Make final X_train_df and X_test_df with constants added
X_train_df = sm.add_constant(X_train_tf, prepend=False, has_constant='add')
X_test_df = sm.add_constant(X_test_tf, prepend=False, has_constant='add')
display(X_train_df.head(2),X_test_df.head(2))


# # Modeling with Statsmodels OLS

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


# ## The Assumptions of Linear Regression

# - The 4 Assumptions of a Linear Regression are:
#     - Linearity: That the input features have a linear relationship with the target.
#     - Independence of features (AKA Little-to-No Multicollinearity): That the features are not strongly related to other features.
#     - **Normality: The model's residuals are approximately normally distributed.**
#     - **Homoscedasticity: The model residuals have equal variance across all predictions.**
# 

# ### QQ-Plot for Checking for Normality

# In[40]:


## Create a Q-QPlot

# first calculate residuals 
resid = y_test - y_hat_test

## then use sm's qqplot
fig, ax = plt.subplots(figsize=(6,4))
sm.graphics.qqplot(resid,line='45',fit=True,ax=ax);


# ### Residual Plot for Checking Homoscedasticity

# In[41]:


## Plot scatterplot with y_hat_test vs resids
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(y_hat_test, resid, ec='white')
ax.axhline(0,c='black',zorder=0)
ax.set(ylabel='Residuals',xlabel='Predicted Revenue')


# ### Putting it all together

# In[42]:


def evaluate_ols(result,X_train_df, y_train, show_summary=True):
    """Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.
    """
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
    


# In[43]:


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
evaluate_ols(result,X_train_df, y_train)


# # Improving Our Model

# ## Functionizing Preprocessing 

# In[44]:


## functionize preprocessing
def prepare_data(df_, drop_cols = [], preview_X=False):
  ## Dropping some features for time
  df = df_.drop(columns=drop_cols)
  ## Treating zipcode as a category
  df['zipcode'] = df['zipcode'].astype(str)

  ## Make x and y variables
  y = df['price'].copy()
  X = df.drop(columns=['price']).copy()

  X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=321)
  ## cat selector
  cat_sel = make_column_selector(dtype_include='object')
  cat_cols = cat_sel(X)
  

  # num selectorr
  num_sel = make_column_selector(dtype_include='number')
  num_cols = num_sel(X)

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
  ## save feature names
  cat_features = preprocessor.named_transformers_['pipeline-2'][-1].get_feature_names_out(cat_cols)
  final_features = [*num_cols,*cat_features]
  X_train_tf = pd.DataFrame( preprocessor.transform(X_train), 
                          columns=final_features, index=X_train.index)
  X_test_tf = pd.DataFrame( preprocessor.transform(X_test),
                         columns=final_features, index=X_test.index)

  ## Make final X_train_df and X_test_df with constants added
  X_train_df = sm.add_constant(X_train_tf, prepend=False, has_constant='add')
  X_test_df = sm.add_constant(X_test_tf, prepend=False, has_constant='add')

  if preview_X:
    display(X_train_df.head(3))
    # X_train_df.info()
  return X_train_df, X_test_df, y_train, y_test


# In[45]:


##import statsmodels correctly
import statsmodels.api as sm


# In[46]:


X_train_df, X_test_df, y_train, y_test = prepare_data(df, preview_X=True)


# # Baseline Model

# In[47]:


X_train_df, X_test_df, y_train, y_test = prepare_data(df, preview_X=True)


## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
evaluate_ols(result,X_train_df, y_train)


# # Removing Outliers

# In[48]:


from scipy import stats
def find_outliers_Z(data, verbose=True):
    outliers = np.abs(stats.zscore(data))>3
    
    
    if verbose:
        print(f"- {outliers.sum()} outliers found in {data.name} using Z-Scores.")

    outliers = pd.Series(outliers, index=data.index, name=data.name)
    return outliers

def find_outliers_IQR(data, verbose=True):
    q3 = np.quantile(data,.75)
    q1 = np.quantile(data,.25)

    IQR = q3 - q1
    upper_threshold = q3 + 1.5*IQR
    lower_threshold = q1 - 1.5*IQR
    
    outliers = (data<lower_threshold) | (data>upper_threshold)
    if verbose:
        print(f"- {outliers.sum()} outliers found in {data.name} using IQR.")
    outliers = pd.Series(outliers, index=data.index, name=data.name)
    return outliers


# In[49]:


num_cols = df.select_dtypes('number').columns
num_cols


# In[50]:


## Loop to remove outliers from same clumns using new function
outlier_cols = num_cols#['runtime','popularity','revenue']

## Empty dict for both types of outliers
outliers_z = {}
outliers_iqr = {}

## Use both functions to see the comparison for # of outliers
for col in outlier_cols:
    outliers_col_z = find_outliers_Z(df[col])
    outliers_z[col] = outliers_col_z
    
    outliers_col_iqr = find_outliers_IQR(df[col])
    outliers_iqr[col] = outliers_col_iqr
    print()
    


# ## Model 2 - Outliers Removed via Z-Scores

# In[51]:


# remove_outliers 
df_clean_z = df.copy()
for col, idx_outliers in outliers_z.items():
  try:
    df_clean_z = df_clean_z[~idx_outliers]
  except:
    print(col, len(idx_outliers), len(df_clean_z))
df_clean_z


# In[52]:


X_train_df, X_test_df, y_train, y_test = prepare_data(df_clean_z)
## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
evaluate_ols(result,X_train_df,y_train)


# ## Model 3 - Outliers Removed IQR Rule

# In[53]:


# remove_outliers
df_clean_iqr = df.copy()
for col, idx_outliers in outliers_iqr.items():
    df_clean_iqr = df_clean_iqr[~idx_outliers]
df_clean_iqr


# In[54]:


X_train_df, X_test_df, y_train, y_test = prepare_data(df_clean_iqr)
## instantiate an OLS model WITH the training data.
model = sm.OLS(y_train, X_train_df)

## Fit the model and view the summary
result = model.fit()
evaluate_ols(result,X_train_df,y_train)


# # Multicollinearity 

# In[55]:


zip_cols = [c for c in X_train_df.columns if c.startswith('zipcode')]


# In[56]:


## Calculating the mask to hide the upper-right of the triangle
plt.figure(figsize=(15,15))
corr = X_train_df.drop(columns=zip_cols).corr().abs()
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr,square=True, cmap='Reds', annot=True, mask=mask);


# In[57]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
 
# separate just x-data and subtract mean
features = X_train_df -  X_train_df.mean()

features


# In[58]:


# create a list of VIF scores for each feature in features.
vif_scores = [variance_inflation_factor(features.values, i) for i in range(len(features.columns))]

# create a new dataframe to hold the VIF scores 
VIF = pd.Series(vif_scores, index=features.columns)
VIF


# In[59]:


## Set float format to view vals not in scientfic notation
pd.set_option('display.float_format',lambda x: f'{x:.2f}')


# In[60]:


## remove infinite values from VIF and sort
VIF = VIF[VIF!=np.inf].sort_values()
VIF


# In[61]:


## filter for VIF that are > 5
VIF[VIF>5]


# In[62]:


high_vif = VIF[VIF>5].index
high_vif


# In[63]:


## Get train data performance from skearn to confirm matches OLS
y_hat_train = result.predict(X_train_df)
print(f'Training R^2: {r2_score(y_train, y_hat_train):.3f}')

## Get test data performance
y_hat_test = result.predict(X_test_df)
print(f'Testing R^2: {r2_score(y_test, y_hat_test):.3f}')


# ### Visualizing Coeffiicents

# In[64]:


result.params


# In[65]:


plt.figure(figsize=(5,6))
ax =result.params.drop([*zip_cols,'const']).sort_values().plot(kind='barh')
ax.axvline()


# In[66]:


plt.figure(figsize=(5,16))
ax =result.params.loc[zip_cols].sort_values().plot(kind='barh')
ax.axvline()


# In[ ]:




