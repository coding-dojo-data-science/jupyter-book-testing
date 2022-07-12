#!/usr/bin/env python
# coding: utf-8

# # Analyzing Coefficients - V2
# - `RENAME: we started to analyze the coefficents in lesson 01-v2. Consider a new name for this like "iterating on our coefficients" or "thoughtful selection of coefficients",etc`

# ## Lesson Objectives

# By the end of this lesson, students will be able to:
# - Extract and visualize coefficients in more helpful formats.
# - ***Interpret coefficients for raw data vs scaled data.***
# - Use coefficient values to inform modeling choices (for insights).
# - Encode nominal categories as ordinal (based on the target)
# - Determine which version of the coefficients would be best for extracting insights and recommendations for a stakeholder. 
# 

# # Our Previous Results

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


## Reviewing the options used
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
pd.set_option('display.float_format', lambda x: f"{x:,.2f}")

## Customization Options
plt.style.use(['fivethirtyeight','seaborn-talk'])
mpl.rcParams['figure.facecolor']='white'

## additional required imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import metrics
import joblib


# ## Code/Model From Previous Lesson

# In[2]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

## Customization Options
plt.style.use(['fivethirtyeight','seaborn-talk'])
mpl.rcParams['figure.facecolor']='white'

## additional required imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import metrics

SEED = 321
np.random.seed(SEED)


# In[3]:


## Load in the King's County housing dataset and display the head and info
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS6xDKNpWkBBdhZSqepy48bXo55QnRv1Xy6tXTKYzZLMPjZozMfYhHQjAcC8uj9hQ/pub?output=xlsx"

df = pd.read_excel(url,sheet_name='student-mat')
df.info()
df.head()


# In[ ]:





# In[4]:


# ## Load in the King's County housing dataset and display the head and info
# df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSEZQEzxja7Hmj5tr5nc52QqBvFQdCAGb52e1FRK1PDT2_TQrS6rY_TR9tjZjKaMbCy1m5217sVmI5q/pub?output=csv")

# ## Dropping some features for time
# df = df.drop(columns=['date'])


# ## Make the house ids the index
# df = df.set_index('id')

# ## drop lat/long
# df = df.drop(columns=['lat','long'])
# ## Treating zipcode as a category
# df['zipcode'] = df['zipcode'].astype(str)

# df.info()
# df.head()


# In[5]:


# ### Train Test Split
## Make x and y variables
y = df['G3'].copy()
X = df.drop(columns=['G3']).copy()

## train-test-split with random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=SEED)


# ### Preprocessing + ColumnTransformer

## make categorical & numeric selectors
cat_sel = make_column_selector(dtype_include='object')
num_sel = make_column_selector(dtype_include='number')

## make pipelines for categorical vs numeric data
cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
                                       fill_value='MISSING'),
                         OneHotEncoder(drop='if_binary', sparse=False))

num_pipe = make_pipeline(SimpleImputer(strategy='mean'))

## make the preprocessing column transformer
preprocessor = make_column_transformer((num_pipe, num_sel),
                                       (cat_pipe,cat_sel),
                                      verbose_feature_names_out=False)

## fit column transformer and run get_feature_names_out
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(preprocessor.transform(X_train), 
                          columns = feature_names, index = X_train.index)


X_test_df = pd.DataFrame(preprocessor.transform(X_test), 
                          columns = feature_names, index = X_test.index)
X_test_df.head(3)


# In[6]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
def evaluate_linreg(model, X_train,y_train, X_test,y_test, return_df=False,
                    get_coeffs=True, coeffs_name = "Coefficients"):

    results = []
    
    y_hat_train = model.predict(X_train)
    r2_train = r2_score(y_train,y_hat_train)
    rmse_train = mean_squared_error(y_train,y_hat_train, squared=False)
    results.append({'Data':'Train', 'R^2':r2_train, "RMSE": rmse_train})
    
    y_hat_test = model.predict(X_test)
    r2_test = r2_score(y_test,y_hat_test)
    rmse_test = mean_squared_error(y_test,y_hat_test, squared=False)
    results.append({'Data':'Test', 'R^2':r2_test, "RMSE": rmse_test})
    
    results_df = pd.DataFrame(results).round(3).set_index('Data')
    results_df.loc['Delta'] = results_df.loc['Test'] - results_df.loc['Train']
    results_df = results_df.T
    

    print(results_df)
        
    if get_coeffs:
        coeffs = pd.Series(model.coef_, index= X_train.columns)
    if model.intercept_!=0:
        coeffs.loc['intercept'] = model.intercept_
    coeffs.name = coeffs_name
    return coeffs


# In[7]:


from sklearn.linear_model import LinearRegression

## fitting a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_df, y_train)
coeffs_orig = evaluate_linreg(lin_reg, X_train_df, y_train, X_test_df,y_test,
                             coeffs_name='Original')
coeffs_orig


# # Iterating On Our Model

# ## Removing the Intercept

# > First, we can remove the intercept from our model, which will force the LinearRegression to explain all of the price without being free to calculate whatever intercept would help the model.
# 

# In[8]:


## fitting a linear regression model
lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(X_train_df, y_train)
coeffs_no_int = evaluate_linreg(lin_reg, X_train_df, y_train, X_test_df,y_test,
                               coeffs_name='No Intercept')
coeffs_no_int.sort_values()


# ### To Intercept or Not To Intercept?

# In[9]:


compare = pd.concat([coeffs_orig, coeffs_no_int],axis=1)
compare = compare.sort_values('Original')
compare['Diff'] = compare['No Intercept'] - compare['Original']
compare


# - At this point, there is a valid argument for using either model as the basis for our stakeholder recommendations. 
# - As long as you are comfortable explaining the intercept as the baseline house price (when all Xs are 0), then it is not difficult to express the findings to a stakeholder.
# - Let's see if either version looks better when visualzied.
# 

# In[10]:


fig, axes = plt.subplots(ncols=2,figsize=(10,8),sharey=True)

compare['Original'].plot(kind='barh',color='blue',ax=axes[0],title='Coefficients + Intercept')
compare['No Intercept'].plot(kind='barh',color='green',ax=axes[1],title='No Intercept')

[ax.axvline(0,color='black',lw=1) for ax in axes]
fig.tight_layout()


# In[11]:


compare['Diff'].plot(kind='barh',figsize=(4,10),color='red',title='Coeff Change W/out Intercept');


# - We can see that by removing the intercept from our model, which had a value of -.95, we have changed the value of several, but not all of the other coefficients.
# - Notice that, in this case, when our model removed a negative baseline value (the intercept), that many of the other coefficients became had a negative change. While this will not always be the case, it does demonstrate how our model has to change the coefficients values when it no longer can calculate a starting grade before factoring in the features.

# ## Scaling Our Features

# - Since we have entirely numeric features, we can simply scale our already-processed X_train/X_test variables by creating a new scaler. 
#     - Note: for more complicated datasets, we would want to create a new precprocessor where we add the scaler to the numeric pipeline.

# In[12]:


# ### Preprocessing + ColumnTransformer
num_pipe_scale = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())

## make the preprocessing column transformer
preprocessor_scale = make_column_transformer((num_pipe_scale, num_sel),
                                       (cat_pipe,cat_sel),
                                      verbose_feature_names_out=False)

## fit column transformer and run get_feature_names_out
preprocessor_scale.fit(X_train)
feature_names = preprocessor_scale.get_feature_names_out()

X_train_scaled = pd.DataFrame(preprocessor_scale.transform(X_train), 
                          columns = feature_names, index = X_train.index)


X_test_scaled = pd.DataFrame(preprocessor_scale.transform(X_test), 
                          columns = feature_names, index = X_test.index)
X_test_scaled.head(3)


# In[13]:


## fitting a linear regression model
lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(X_train_scaled, y_train)
coeffs_scaled = evaluate_linreg(lin_reg, X_train_scaled, y_train, X_test_scaled,y_test)
coeffs_scaled


# In[14]:


fig, ax = plt.subplots(figsize=(5,8))
coeffs_scaled.sort_values().plot(kind='barh')
# compare['Original'].plot(kind='barh',color='blue',ax=axes[0],title='Coefficients + Intercept')
# compare['No Intercept'].plot(kind='barh',color='green',ax=axes[1],title='No Intercept')

ax.axvline(0,color='black',lw=1)


# # 📌 TO DO
# - visualize and discuss the scaled coefficients
# - select scaled vs not scaled

# # Revisiting Our Business Case

# - Thus far, we have done all of our modeling under the assumption that we want to predict how well current students will do in their final year.
# - However, the stakeholder likely cares more about identifying how students will perform at very beginning of their Year 1. 
#     - Let's keep this in mind and remove any features that we would not have known when the student was at the beginning of Year 1.

# ## Modeling - For New Students

# - We **must** remove:
#     - G1: We wouldn't know year 1 grades yet.
#     - G2: We wouldn't know year 1 grades yet.
#     
# - We should **probably** remove:
#     - paid: We would not know if students paid for extra classes in the subject yet. 
#         - Though we may be able to find out if they are WILLING to pay for extra classes.
#     - activities: We would not know if the student was involved in extracurriculars at this school yet.
#         - Though we may be able to ask students if they INTEND to be involved in activities.
#         
# 
# - We **may** want to remove:
#     - absences:
#         - We wouldn't have absences from the current school, though we likely could get absences from their previous school.
#     - Dalc: Work day alcohol consumption. Hopefully, the students who have not entered high school yet will not already be consuming alcohol.
#     - Walc: weekend alcohol consumption. Hopefully, the students who have not entered high school yet will not already be consuming alcohol.
# 
# As you can see, some of the features are obviously inappropriate to include, but many of them are a bit more ambiguous. 
#     - Always think of your stakeholder's goals/problem statement when deciding what features to include in your model/analysis. 
# >- When in doubt, contact and ask your stakeholder about the choice(s) you are considering!

# ### `DECIDE IF USING SCALED`

# #### Unsaled

# In[15]:


## remove cols that MUST be removed.

df_mvp = df.drop(columns=['G1','G2'])

# ### Train Test Split
## Make x and y variables
y = df_mvp['G3'].copy()
X = df_mvp.drop(columns=['G3']).copy()

## train-test-split with random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=SEED)



## fit column transformer and run get_feature_names_out
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(preprocessor.transform(X_train), 
                          columns = feature_names, index = X_train.index)


X_test_df = pd.DataFrame(preprocessor.transform(X_test), 
                          columns = feature_names, index = X_test.index)
X_test_df.head(3)


# In[16]:


## fitting a linear regression model
lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(X_train_df, y_train)
coeffs_mvp = evaluate_linreg(lin_reg, X_train_df, y_train, X_test_df,y_test)
coeffs_mvp


# - As we can see above, NOT including the grade from year 2 dramatically hurts our model's ability to predict the final grade. 

# #### Scaled

# In[17]:


## fit column transformer and run get_feature_names_out
preprocessor_scale.fit(X_train)
feature_names = preprocessor_scale.get_feature_names_out()

X_train_scaled = pd.DataFrame(preprocessor_scale.transform(X_train), 
                          columns = feature_names, index = X_train.index)


X_test_scaled = pd.DataFrame(preprocessor_scale.transform(X_test), 
                          columns = feature_names, index = X_test.index)
X_test_df.head(3)


# In[18]:


## fitting a linear regression model
lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(X_train_scaled, y_train)
coeffs_mvp_scaled = evaluate_linreg(lin_reg, X_train_scaled, y_train,
                                    X_test_scaled,y_test, 
                                    coeffs_name="Scaled Coefficients")
coeffs_mvp_scaled


# ### Final Comparison

# In[19]:


compare = pd.concat([coeffs_mvp, coeffs_mvp_scaled],axis=1)
compare


# In[20]:


compare = compare.sort_values('Coefficients')
compare['Diff'] = compare['Scaled Coefficients'] - compare['Coefficients']
compare


# In[21]:


fig, axes = plt.subplots(ncols=2,figsize=(10,8),sharey=True)

compare['Coefficients'].plot(kind='barh',color='blue',ax=axes[0],title="Raw")
compare['Scaled Coefficients'].plot(kind='barh',color='green',ax=axes[1],title='Scaled')

[ax.axvline(0,color='black',lw=1) for ax in axes]
fig.tight_layout()


# In[22]:


# ax = coeffs_mvp.sort_values().plot(kind='barh',figsize=(6,10))
# ax.axvline(0,color='k')
# ax.set_title('LinearRegression Coefficients');


# - Notice how VERY diferent our coefficients are now that we have removed the  students' grades from the prior 2 years!

# > BOOKMARK:  interpret new coeffs

# In[ ]:





# ## Selecting Our Final Model for Extracting Insights

# - 

# - Out of all of the variants we have tried, the best one to use going forward is our Ordinal encoded zipcodes with an intercept. (once again, you can make an argument for using the one without an intercept as well).
# 

# In[23]:


coeffs_mvp.sort_values()


# - In the next lesson, we will focus on another type of model-based values.
# 
