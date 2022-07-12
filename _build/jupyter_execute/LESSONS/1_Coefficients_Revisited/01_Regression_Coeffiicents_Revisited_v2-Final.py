#!/usr/bin/env python
# coding: utf-8

# # Regression Coefficients - Revisited

# ## Lesson Objectives

# By the end of this lesson, students will be able to:
# - Use scikit-learn v1.1's simplified toolkit.
# - Extract and visualize coefficients from sklearn regression model. 
# - Control panda's display options to facilitate interpretation.
# 

# ## Introduction

# - At the end of last stack, we dove deep into linear regression models and their assumptions. We introduced a new package called statsmodels, which produced a Linear Regression model using "Ordinary-Least-Squared (OLS)". 
# - The model included a robust statistical summary that was incredibly informative as we critically diagnosed our regression model and if we met the assumptions of linear regression.
# - This stack, we will be focusing on extracting insights from our models: both by examining parameters/aspects of the model itself, like the coefficients it calculated, but also by applying some additional tools and packages specifically designed to explain models. 
# 
# - Most of these tools are compatible with the scikit-learn ecosystem but are not yet available for statsmodels.
# 
# Since we are not focusing on regression diagnostics this week, we will shift back to using scikit-learn models. Scikit-learn recently released version 1.1.1, which added several helpful tools that will simplify our workflow. 
# 
# Let's review some of these key updates as we rebuild our housing regression model from week 16.
# 

# # Confirming Package Versions

# - All packages have a version number that indicates which iteration of the package is currently being used.
#     - If you import an entire package, you can use the special method `package.__version__` (replace package with the name of the package you want to check).
# - The reason this is important is that as of the writing of this stack, Google Colab is still using a version of python that is too old to support the newest scikit-learn.
#     - You can check which version of python you are using by running the following command in a jupyter notebook:
#         - `!python --version`
#         - Note: if you remove the `!`, you can run this command in your terminal.
# 
# - If you run the following code on Google Colab and on your local computer, you can compare the version numbers. 
#         
# <img src="colab_versions.png" width=400px>
# 
# - Now, run the following block of code in a jupyter notebook on your local machine to confirm that you have Python 3.8.13 and sklearn v1.1.1.
# 

# In[1]:


# Run the following command on your local computer to 
import sklearn
print(f"sklearn version: {sklearn.__version__}")
get_ipython().system('python --version')


# 
# >- If you have a Python 3.7 or an earlier version of scikit-learn, please revisit the "`<Insert the name of the "week" of content on the LP for installation>`". 
#     - See the "`Updating Your Dojo-Env Lesson` [Note to Brenda: does not exist yet - see 1:1 doc for question on handling multiple envs] for how to remove your current dojo-env and replace it with the new one.

# # Extracting Coefficients from LinearRegression in scikit-learn

# ## Highlighted Changes  - scikit-learn v1.1

# - The single biggest change in the updated sklearn is a fully-functional `.get_feature_names_out()` method in the `ColumnTransformer`.
#     - This will make it MUCH easier for us to extract our transformed data as dataframes and to match up the feature names to our models' coefficients.
# - There are some additional updates that are not pertinent to this stack, but if you are curious, you can find the [details on the new release here](https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_1_0.html).

# ## New and Improved `ColumnTransformer` 

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

# set random seed
SEED = 321
np.random.seed(SEED)

## set text displays for sklearn
from sklearn import set_config
set_config(display='text')


# In[ ]:





# In[3]:


## Load in the student performance - math dataset & display the head and info
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS6xDKNpWkBBdhZSqepy48bXo55QnRv1Xy6tXTKYzZLMPjZozMfYhHQjAcC8uj9hQ/pub?output=xlsx"

df = pd.read_excel(url,sheet_name='student-mat')
df.info()
df.head()


# In[4]:


df.nunique()


# ### Selecting Our Features

# - If we wanted to make recommendations to the school district on how to identify and help students that will perform poorly by year 3, we should think about what features make the most sense to include.
# 
# - There are ~2 different approaches  we could take to what to include, depending on what use case we are addressing for our stakeholder.
# 
# For example, if our primary goal is to just identify 3rd year students that will perform poorly, then including all of these features would make sense.
# 
# However, if our primary goal is to identify which INCOMING students will perform poorly by their 3rd year, then we would NOT include G1 or G2, since the school will not have those grades for bran new incoming students.
# 
# - We will start our analysis addressing the first use case, identify rising 3rd year students that will perform poorly.

# ### Train Test Split

# In[5]:


## Make x and y variables
y = df['G3'].copy()
X = df.drop(columns=['G3']).copy()

## train-test-split with random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=SEED)
X_train.head()


# ### Preprocessing + ColumnTransformer

# In[6]:


## make categorical selector and verifying it works 
cat_sel = make_column_selector(dtype_include='object')
cat_sel(X_train)


# In[7]:


## make numeric selector and verifying it works 
num_sel = make_column_selector(dtype_include='number')
num_sel(X_train)


# In[8]:


## make pipelines for categorical vs numeric data
cat_pipe = make_pipeline(SimpleImputer(strategy='constant',
                                       fill_value='MISSING'),
                         OneHotEncoder(handle_unknown='ignore', sparse=False))

num_pipe = make_pipeline(SimpleImputer(strategy='mean'))


# > Nothing we have done yet should be new code. The changes we will make will be when we create our ColumnTransformer with `make_column_transformer`.
# - From now on, you should add `verbose_feature_names_out=False` to `make_column_transformer`

# In[9]:


## make the preprocessing column transformer
preprocessor = make_column_transformer((num_pipe, num_sel),
                                       (cat_pipe,cat_sel),
                                      verbose_feature_names_out=False)
preprocessor


# >- In order to extract the feature names from the preprocessor, we first have to fit it on the data.
# - Next, we can use the `preprocessor.get_feature_names_out()` method and save the output as something like "feature_names" or "final_features".

# In[10]:


## fit column transformer and run get_feature_names_out
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()
feature_names


# - Notice how we were able to get the complete list of feature names, including the One Hot Encoded features with their proper "zipcode" prefix. 
# - Quick note: if you forgot to add `verbose_feature_names_out` when you made your preprocessor, you would get something like this:
# 

# In[11]:


## make the preprocessing column transformer
preprocessor_oops = make_column_transformer((num_pipe, num_sel),
                                       (cat_pipe,cat_sel)
                                           ) # forgot verbose_feature_names_out=False
## fit column transformer and run get_feature_names_out
preprocessor_oops.fit(X_train)
feature_names_oops = preprocessor_oops.get_feature_names_out()
feature_names_oops


# ### Remaking Our X_train and X_test as DataFrames

# - Now that we have our list of feature names, we can very easily transform out X_train and X_test into preprocessed dataframes. 
# - We can immediately turn the output of our preprocessor into a dataframe and do not need to save it as a separate variable first.
#     - Therefore, in our pd.DataFrame, we will provide the `preprocessor.transform(X_train)` as the first argument, followed by `columns=feature_names` (the list we extracted from our precprocessor)
#     - Pro Tip: you can also use the same index as your X_train or X_test variable, if you want to match up one of the transformed rows with the original dataframe.

# In[12]:


X_train_df = pd.DataFrame(preprocessor.transform(X_train), 
                          columns = feature_names, index = X_train.index)
X_train_df.head(3)


# In[13]:


X_test_df = pd.DataFrame(preprocessor.transform(X_test), 
                          columns = feature_names, index = X_test.index)
X_test_df.head(3)


# In[14]:


## confirm the first 3 rows index in y_test matches X_test_df
y_test.head(3)


# - Notice that we cannot see all of our features after OneHotEncoding. Pandas truncates the display in the middle and displays `...` instead. 
# - We can get around this by changing the settings in Pandas using `pd.set_option`
#     - In this case, we want to change the `max_columns` to be a number larger than our number of final features. Since we have 87 features, setting the `max_columns` to 100 would be sufficient.
# - For more information on pandas options, see their [documentation on Options and Settings](https://pandas.pydata.org/docs/user_guide/options.html)
# - Final note: in your project notebooks, you should add this function to the top of your notebook right after your imports.

# In[15]:


## Using pd.set_option to display more columns
pd.set_option('display.max_columns',100)
X_train_df.head(3)


# ## Extracting Coefficients and Intercept from Scikit-Learn Linear Regression

# In[16]:


from sklearn.linear_model import LinearRegression

## fitting a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_df, y_train)
print(f'Training R^2: {lin_reg.score(X_train_df, y_train):.3f}')
print(f'Test R^2: {lin_reg.score(X_test_df, y_test):.3f}')


# In[17]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
def evaluate_linreg(model, X_train,y_train, X_test,y_test, return_df=False):

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
    
    if return_df:
        return results_df
    else:
        print(results_df)


# In[18]:


## fitting a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_df, y_train)
evaluate_linreg(lin_reg, X_train_df, y_train, X_test_df,y_test)


# - For scikit-learn Linear Regressions, we can find the coefficients for the features that were included in our X-data under the `.coef_` attribute. 
# -  the `.coef_` is a numpy matrix that should have the same number of values as the # of columns in X_train_df

# In[19]:


lin_reg.coef_


# In[20]:


## Checking the number of coeffs matches the # of feature names
print(len(lin_reg.coef_))
len(feature_names)


# > Note: if for some reason the length of your coef_ is 1, you should add the `.flatten()` method to convert the  coef_ into a simple 1-D array.

# ### Saving the coefficients as a pandas Series

# - We can immediately turn the the models' .coef_ into a pd.Series, as well.
#     - Therefore, in our pd.Series, we will provide the `lin_reg.coef_` as the first argument, followed by `index=feature_names` (pandas Series are 1D and do not have columns)

# In[21]:


## Saving the coefficients
coeffs = pd.Series(lin_reg.coef_, index= feature_names)
coeffs


# - The constant/intercept is not included in the .ceof_ attribute (if we used the default settings for LinearRegression which sets fit_intercept = True)
# - The intercept is stored in the `.intercept_` attribute 
# - We can add this as a new value to our coeffs series.
# - Note: it is up to you what you name your intercept/constant. If you wanted to keep the naming convention of statsmodels, you could use "const" or just "intercept" for simplicity.

# In[22]:


# use .loc to add the intercept to the series
coeffs.loc['intercept'] = lin_reg.intercept_
coeffs


# ### Displaying the Coefficients

# - Just like we increased the number of columns displayed by pandas, we can also increase the number of rows displayed by pandas.
# - CAUTION: DO NOT SET THE MAX ROWS TO 0!! If you try to display a dataframe that has 1,000,000 it will try to display ALL 1,000,000 rows and will crash your kernel.

# In[23]:


# pd.set_option('display.max_rows',100)
coeffs


# ### Suppressing Scientific Notation in Pandas

# > We can ALSO use panda's options to change how it display numeric values.
# - if we want to add a `,` separator for thousands and round to 2 decimal places, we would use the format code ",.2f". 
# - In order for Pandas to use this, we will have to use an f-string with a lambda x. (X represent any numeric value being displayed by pandas).

# In[24]:


pd.set_option('display.float_format', lambda x: f"{x:,.2f}")
coeffs


# ## Inspecting Our Coefficients - Sanity Check

# - Hmmmm....hold on now. We saw last lesson that our target, G3, contained scores from 0 to 20. 
# - So HOW IN THE WORLD does it make sense that your model's baseline value (the y-intercept) is 2 trillion?!?
# - This may be due to us introducing multicollinearity during One Hot Encoding.

# #### Using OneHotEncoder with binary categorical features.

# - if we check just our string columns for the # of unique values:

# In[25]:


df.select_dtypes('object').nunique()


# - We can see that many of our categories only have 2 options.
#     - For example: the paid feature.
#         - One Hot Encoding this feature will create a "paid_no" column and "paid_yes" column.
# 
# - Here is where we should think about our final use case for this data. If we want to explain student performance, there is no benefit to one-hot-encoding both categories. 
#     - We know that if someone has a 0 for "paid_yes" that it means "paid_no" would be 1. 
#     
#     
# - To remove these unnecessary columns, we can change our arguments for our OneHotEncoder in our pipeline and add "`drop='if_binary'.
#     - HOWEVER, we cannot use BOTH `handle_unknown` AND the `drop` argument together. We will get an error message.
# 
# - Since our current modeling will be used to extract insights for our stakeholder and will not be deployed to the cloud where it will run predictions on new data, we can safely switch to using the drop='if_binary' option.

# ## Recreating Our X/y data with `drop='if_binary'`

# In[26]:


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


# - As we can see above, we now only have 1 version of our binary columns (e.g. "paid_yes","internet_yes")

# ## Refitting a LinearRegression

# In[27]:


## fitting a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_df, y_train)
evaluate_linreg(lin_reg, X_train_df, y_train, X_test_df,y_test)


# In[28]:


## Saving the coefficients
coeffs = pd.Series(lin_reg.coef_, index= feature_names)
coeffs['intercept'] = lin_reg.intercept_
coeffs


# - OK, now THESE coefficients pass our sanity check. They are all values <1, which seems appropriate for predicting a value between 0 and 100.

# ### VIsualizing Coefficients

# - Now, let's examine the coefficients below and see if they make sense, based on our knowledge about houses. 
# 
# 
# 
# - To more easily analyze our coefficients, we will sort them by value and plot them as a horizontal bar graph.
# - Let's also load in the data dictionary so we can make informed interpretations about the meaning of our features

# In[29]:


## Plot the coefficients
ax = coeffs.sort_values().plot(kind='barh',figsize=(6,10))
ax.axvline(0,color='k')
ax.set_title('LinearRegression Coefficients');


# - **Attributes for both student-mat.csv (Math course) and student-por.csv (Portuguese language course) datasets:**
# - 1 school - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
# - 2 sex - student's sex (binary: "F" - female or "M" - male)
# - 3 age - student's age (numeric: from 15 to 22)
# - 4 address - student's home address type (binary: "U" - urban or "R" - rural)
# - 5 famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
# - 6 Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
# - 7 Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# - 8 Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# - 9 Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
# - 10 Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
# - 11 reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
# - 12 guardian - student's guardian (nominal: "mother", "father" or "other")
# - 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# - 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# - 15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# - 16 schoolsup - extra educational support (binary: yes or no)
# - 17 famsup - family educational support (binary: yes or no)
# - 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# - 19 activities - extra-curricular activities (binary: yes or no)
# - 20 nursery - attended nursery school (binary: yes or no)
# - 21 higher - wants to take higher education (binary: yes or no)
# - 22 internet - Internet access at home (binary: yes or no)
# - 23 romantic - with a romantic relationship (binary: yes or no)
# - 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# - 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# - 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# - 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# - 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# - 29 health - current health status (numeric: from 1 - very bad to 5 - very good)
# - 30 absences - number of school absences (numeric: from 0 to 93)
# 
# 
# 
# - These grades are related with the course subject, Math or Portuguese:
#     - 31 G1 - first period grade (numeric: from 0 to 20)
#     - 31 G2 - second period grade (numeric: from 0 to 20)
#     - 32 G3 - final grade (numeric: from 0 to 20, output target)
# 
# - Additional note: 
#     - there are several (382) students that belong to both datasets . 
#     - These students can be identified by searching for identical attributes
#     - that characterize each student, as shown in the annexed R file.
# 

# ### Reminder: Interpreting Coefficients
# 
# When interpreting coefficients we need to keep in mind what transformations have been applied to our features. 
# - If its a numeric feature and it has NOT been scaled:
#     - The coefficient tells us "when I increase the value for this feature by **1 unit**, this is the change in the target."
# - If its a numeric feature that has been scaled using Z-scores/StandardScaler:
#     - The coefficient tells us "when I increase the value for this feature by **1 Standard Deviation** this is the change in the target." 
# - If its a One-Hot Encoded categorical feature:
#     - The coefficient tells us "if you belong to this category, this will be the change in the target" 

# #### Interpreting Our Model's Coefficients

# - So, looking at our coefficient plot above and the Series below, let's interpret the 3 largest coefficients that increase G3, and the largest 3 that decrease G3 .

# In[30]:


coeffs.sort_values()


# - Coefficients that Positively Influence Final Grade:
#     - G2 (Year 2 Grade): 
#         - Increasing their Year 2 grade by 1 will also increase their predict G3 by 0.99 points.
#     - higher_yes (student intends to continue higher education):
#         - Being in the yes group (wanting hire education) increases their performance by 0.82
#     - For Pstatus_T (Parents live Together):
#         - Having parents living together increases their final grade by 0.38 points.
#         
#         
# - Coefficients that Negatively Influence Final Grade:
#     - Intercept:
#         - Our model assumed a starting score of -.95.
#     - Fjob_services:
#         - Having a father with a job in administration/police  subtracts 0.54 points from G3.
#     - reason_home: 
#         - Choosing the school because it is simply close to home subtracts 0.42 points from G3.
#     - Activities_yes:
#         - Being involved in Extracurricular activities decreases G3 by .34 points.
#         
#    

# 
# #### `<DEBATING WHETHER TO INCLUDE FINAL STORY-TELLING SUMMARY OR NOT> `
# - Overall, these coefficients make a lot of intuitive sense!
#     - Students who have a history of good grades, who plan to continue their education, and who have an intact nuclear family will `perform well<rephase>` in their final year.
#     - Students that are attending this school due because its close to home, who are involved in extracurricular activities will have reduced grades for their final year.
# - However, it is a little harder to understand how having a father with a job in "services" (administration or police) causes a decrease in grades. 
#     - Let's keep this coefficient in mind as we move forward and iterate on our regression modeling.

# ### Next Steps

# - In the next lesson, we will iterate upon this model and discuss selecting a final regression model with coefficients that are appropriate for our purposes. 

# ## Summary

# - In this lesson, we revisited linear regression with scikit-learn. We introduced some simplifications to our workflow and discussed extracting coefficients from our LinearRegression model. 
# 
# - Next lesson we will iterate on our current model to find more intuitive coefficients that we can use to extract insight for our stakeholders.

# ### Recap - Sklearn v1.1

# - We added the argument `verbose_feature_names_out=False` to `make_column_transformer`, which let us extract our feature names (after fitting the preprocessor) using `.get_feature_names_out()`
# 
# - We then used this list of features when reconstruction our transformed X_train and X_test as dataframes and when extracting coefficients from our model.

# ### Recap - Pandas Options

# - We used the following options in our notebook. Ideally, we should group these together and move them to the top of our notebook, immediately after our imports.

# In[31]:


## Reviewing the options used
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
pd.set_option('display.float_format', lambda x: f"{x:,.2f}")

