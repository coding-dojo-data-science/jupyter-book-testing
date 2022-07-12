#!/usr/bin/env python
# coding: utf-8

# # Model Explainers - For Classification

# - TO DO:
# - This will come after the lesson on converting regression task to classification
# 

# ## ADMIN: References
# - REF: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/NHANES%20I%20Survival%20Model.html

# ## Lesson Objectives

# By the end of this lesson students will be able to:
# - Define a global vs local explanation
# - Use the Shap package and interpret shap values.
# 

# ## Model Explainers

# - There are packages with the sole purpose of better understanding how machine learning models make their predictions.
# - Generally, model explainers will take the model and some of your data and apply some iterative process to try to quantify how the features are influencing the model's output.

# In[ ]:





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
SEED = 321
np.random.seed(SEED)


# In[2]:


## Adding folder above to path
import os, sys
sys.path.append(os.path.abspath('../../'))

## Load stack_functions with autoreload turned on
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from CODE import stack_functions as sf
from CODE import prisoner_project_functions as pf

def show_code(function):
    import inspect 
    from IPython.display import display,Markdown, display_markdown
    code = inspect.getsource(function)
    md_txt = f"```python\n{code}\n```"
    return display(Markdown(md_txt))
    


# In[3]:


# show_code(pf.evaluate_classification)


# ## DATASET - `NEED TO FINALIZE`

# ### Preprocessing Titanic

# In[4]:


## Load in the King's County housing dataset and display the head and info
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS6xDKNpWkBBdhZSqepy48bXo55QnRv1Xy6tXTKYzZLMPjZozMfYhHQjAcC8uj9hQ/pub?output=xlsx"

df = pd.read_excel(url,sheet_name='student-por')
# df.drop(columns=['G1','G2'])
df.info()
df.head()


# ## QUICK CONVERT TO CLASS

# - Maybe curve grades? 
#     - https://www.wikihow.com/Curve-Grades
# - Or just make letter grades?
# 
# 

# In[5]:


grade_cols = ['G1','G2','G3']
for col in grade_cols:
    df[f"{col}(%)"] = (df[col]/20) *100
# df[['G1%','G2%','G3%']]  = df[['G1','G2','G3']]/20*100
df


# In[6]:


x = df['G3(%)'].values
x


# In[7]:


def calc_letter_grade(x):
    if isinstance(x,pd.Series):
        x = x.values
    letter_grades = {'A':x>=90,
                'B':(80<=x)&(x<90),
                'C':(70<=x)&(x<80),
                'D':(60<=x)&(x<70),
                'F':x<60}
    return np.select(letter_grades.values(), letter_grades.keys())


# In[8]:


calc_letter_grade(df['G3(%)'])


# In[9]:


# letter_grades = {'A':x>=90,
#                 'B':(80<=x)&(x<90),
#                 'C':(70<=x)&(x<80),
#                 'D':(60<=x)&(x<70),
#                 'F':x<60}
# np.select(letter_grades.values(), letter_grades.keys())


# In[10]:


grade_cols_perc = [f"{col}(%)" for col in grade_cols]
df[grade_cols_perc]


# In[11]:


for col in grade_cols_perc:
    df[col.replace("(%)",'_Class')] = calc_letter_grade(df[col])
df


# In[12]:


fig, axes = plt.subplots(nrows=2,figsize=(8,8))
sns.histplot(data=df, x='G3(%)',ax=axes[0], binwidth=10)

sns.countplot(data=df,x='G3_Class',ax=axes[1],order=['F','D','C','B','A'])


# In[13]:


df['G3_Class'].value_counts()


# In[14]:


## Define target as had a F or Above
df['target_F'] = df['G3_Class'] == 'F'
df['target_F'].value_counts()


# In[15]:


g_cols = [c for c in df.columns if c.startswith("G")]
g_cols


# 

# In[16]:


# ### Train Test Split
## Make x and y variables
drop_feats = [*g_cols]
y = df['target_F'].copy()
X = df.drop(columns=['target_F',*drop_feats]).copy()

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


# In[17]:


## fit random fores
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_df,y_train)
sf.evaluate_classification(rf_clf,X_test_df,y_test, 
                       X_train=X_train_df, y_train=y_train)#linreg(rf_reg,X_train_zips,y_train,X_test_zips,y_test)


# ### Loading Joblib of Regressions from Lesson 04

# In[18]:


# import joblib
# ## If showing joblib in prior lesson, this cell will be included and further explained
# loaded_data = joblib.load("../4_Feature_Importance/lesson_04.joblib")
# loaded_data.keys()


# In[19]:


# ## If showing joblib in prior lesson, this cell will be included and further explained
# X_train_reg = loaded_data['X_train'].copy()
# y_train_reg = loaded_data['y_train'].copy()
# X_test_df_reg = loaded_data['X_test'].copy()
# y_test_reg = loaded_data['y_test'].copy()
# lin_reg = loaded_data['lin_reg']
# rf_reg = loaded_data['rf_reg']


# ## Using SHAP for Model Interpretation

# In[ ]:





# - SHAP (SHapley Additive exPlanations)) 
#     - [Repository](https://github.com/slundberg/shap)
#     - [Documentation](https://shap.readthedocs.io/en/latest/?badge=latest)
#   
# 
# - SHAP uses game theory to calcualte Shapely values for each feature in the dataset. 
# - Shapely values are calculated by iteratively testing each feature's contribution to the model by comparing the model's  performance with vs. without the feature. (The "marginal contribution" of the feature to the model's performance).

# #### Papers, Book Excerpts, and  Blogs
# - [White Paper on Shapely Values](https://arxiv.org/abs/1705.07874)
#     
# - [Intepretable Machine Learning Book - Section on SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)
#     
# - Towards Data Science Blog Posts:
#     - [Explain Your Model with SHAP Values](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d)
# 
#     - [Explain Any Model with SHAP KernelExplaibner](https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a)

# #### Videos/Talks:
# - Explaining Machine Learning Models (in general).
#     - ["Open the Black Box: an intro to Model Interpretability with LIME and SHAP](https://youtu.be/C80SQe16Rao)
# - Understanding Shapely/SHAP Values:
#     - [AI Simplified: SHAP Values in Machine Learning ](https://youtu.be/Tg8aPwPPJ9c)- (Intuitive Explanation)
#     - [Explainable AI explained! | #4 SHAP  ](https://youtu.be/9haIOplEIGM)- (Math Calculation Explanation)

# ### How To Use Shap

# 
# - Import and initialize javascript:
# 
# ```python
# import shap 
# shap.initjs()
# ```

# In[20]:


import shap
shap.initjs()


# ### Shap Explainers

# - shap has several types of model explainers that are optimized for different types of models. 
# 
# 
# #### Explainers and their use cases:
# 
# 
# | Explainer                         | Description                                                                                    |
# |:----------------------------------|:-----------------------------------------------------------------------------------------------|
# | shap.Explainer                    | Uses Shapley values to explain any machine learning model or python function.                  |
# | shap.explainers.Tree              | Uses Tree SHAP algorithms to explain the output of ensemble tree models.                       |
# | shap.explainers.Linear            | Computes SHAP values for a linear model, optionally accounting for inter-feature correlations. |
# | shap.explainers.Permutation       | This method approximates the Shapley values by iterating through permutations of the inputs.   |
# | shap.explainers.Sampling          | This is an extension of the Shapley sampling values explanation method (aka.                   |
# | shap.explainers.Additive          | Computes SHAP values for generalized additive models.                                          |
# | shap.explainers.other.Coefficent  | Simply returns the model coefficents as the feature attributions.                              |
# | shap.explainers.other.Random      | Simply returns random (normally distributed) feature attributions.                             |
# | shap.explainers.other.LimeTabular | Simply wrap of lime.lime_tabular.LimeTabularExplainer into the common shap interface.          |
# | shap.explainers.other.Maple       | Simply wraps MAPLE into the common SHAP interface.                                             |
# | shap.explainers.other.TreeMaple   | Simply tree MAPLE into the common SHAP interface.                                              |
# | shap.explainers.other.TreeGain    | Simply returns the global gain/gini feature importances for tree models.                       |
# 
# <!-- - Uses game theory to explain feature importance and how a feature steered a model's prediction(s) by removing each feature and seeing the effect on the error.
# 
# - SHAP has:
#     - `TreeExplainer`:
#         - compatible with sckit learn, xgboost, Catboost
#     - `KernelExplainer`:
#         - compatible with "any" model
#          -->
# 
# 
# - See [this blog post](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d) for intro to topic and how to use with trees
# 
# - For non-tree/random forest models [see this follow up post]( https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a)
# 
#         

# ### Preparing Data for Shap

# - Shap's approach to explaining models can be very resource-intensive for complex models such as our RandomForest.
# - To get around this issue, shap includes a convenient smapling function to save a small sample from one of our X variables.

# In[21]:


X_shap = shap.sample(X_train_df,nsamples=200,random_state=321)
X_shap


# In[22]:


## get the corresponding y-values
y_shap = y_train.loc[X_shap.index]
y_shap


# ### Explaining Our RandomForest

# 
# 1. Create a shap explainer using your fit model.
# 
# ```python
# explainer = shap.TreeExplainer(rf_reg)
# ```
# 
# 2. Get shapely values from explainer for your training data
# 
# ```python
# shap_values = explainer(X_shap)
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

# In[23]:


# # TEMP
# X_shap = shap.sample(X_train_df,nsamples=200,random_state=321)

# explainer = shap.TreeExplainer(rf_reg)
# shap_values_demo = explainer.shap_values(X_shap,y_shap)
# shap_values_demo[1]


# In[24]:


X_train_df


# In[25]:


# X_shap = shap.sample(X_train_df,nsamples=200,random_state=SEED)
X_shap = X_train_df.copy()
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer(X_shap,y_shap)
shap_values[0]


# In[26]:


X_shap.shape


# In[27]:


shap_values.shape


# - We can see that shap calculated values for every row/column in our X_shap variable.
# - What does the first row's shap values look like?

# In[28]:


shap_values[0]


# - Notice above that we do not seem to have a simple numpy array. 

# In[29]:


type(shap_values[0])


# In[30]:


explanation_0 = shap_values[0]
explanation_0


# In[ ]:





# - Each entry in the shap_values array is new type of object called an Explanation.
#     - Each Explanation has:
#         - values:the shap values calculated for this observation/row. 
#             - For classification models, there is a column with values for each target.
#         - base_values: the final shap output value
#         - data: the original input feature

# In[31]:


## Showing .data is the same as the raw X_shap
explanation_0.data


# In[32]:


X_shap.iloc[0].values


# In[33]:


## showing the .values
pd.DataFrame(explanation_0.values,index=X_shap.columns)


# # 📌 **BOOKMARK**

# ## Shap Visualizations - Classification

# ### Summary Plot

# ```python
# ## For normal bar graph of importance:
# shap.summary_plot(shap_values[:,:,1],features=X_shap, plot_type='bar')
# 
# ## For detail Shapely value visuals:
# shap.summary_plot(shap_values[:,:,1], features=X_shap)
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

# - First, let's examine a simple version of our shap values. 
#     - By using the plot_type="bar" version of the summary plot, we get something that looks very similar to the feature importances we discussed previously. 
#     

# In[34]:


shap.summary_plot(shap_values[:,:,1],features= X_shap,plot_type='bar')


# - In this case, it is using the magnitude of the average shap values to to show which features had the biggest impact on the model's predictions.
#     - Like feature importance and permutation importance, this visualization is not indicating which **direction** the features push the predict.
#     
# - Now, let's examine the "dot" version of the summary plot. 
#     - By removing the plot_type argument, we are using the default, which is "dot". 
#         - We could explicitly specify plot_type="dot".
#             - There are also additional plot types that we will not be discussing in this lesson (e.g. "violin","compact_dot")

# In[35]:


# shap.summary_plot(shap_values[:,:,1],X_shap)


# In[36]:


shap.summary_plot(shap_values[:,:,1],X_shap)


# TO DO: "Failures""

# Now THAT is a lot more nuanced of a visualization!
# Let's break down how to interpret the visual above.

# In[ ]:





# In[37]:


# shap.summary_plot(shap_values[:,:,1],features= X_shap,plot_type='compact_dot')


# In[38]:


## violin version.
shap.summary_plot(shap_values[:,:,1],features= X_shap,plot_type='violin')


# ### Dependence Plots

# 
# Shap also includes the `shap.dependence_plot`
# which show how the model output varies by a specific feature. By passing the function a feature name, it will automatically determine what features may driving the interactions with the selected feature. It will encode the interaction feature as color.
# ```python
# ## To Auto-Select Feature Most correlated with a specific feature, just pass the desired feature's column name.
# 
# shap.dependence_plot('Age', shap_values[:,:,1], X_shap)
# ```
# 
# - TO DO:
#     - There is a way to specifically call out multiple features but I wasn't able to summarize it quickly for this nb
# ```

# In[39]:


# shap_values[:,:,1].values


# In[40]:


## Using shap_values made from shap_values = explainer(X_shap)
shap.dependence_plot("failures", shap_values[:,:,1].values,X_shap)


# - ?Men are more likely to have failures?

# In[41]:


## Using shap_values made from shap_values = explainer(X_shap)
shap.dependence_plot("sex_M", shap_values[:,:,1].values,X_shap)


# - ?Being male interacts with Weekend alcohol consumption??

# In[42]:


## Using shap_values made from shap_values = explainer(X_shap)
shap.dependence_plot("age", shap_values[:,:,1].values,X_shap)


# - ?The older the student the more likely the reason for this school was because of a specific course?

# ### Force Plot

# >- Note: the force_plot is an interactive visualization that uses javascript. You must Trust your jupyter notebook in order to display it.
#     - In the top right corner of jupyter notebook, next the kernel name (Python (dojo-env)), click the `Not Trusted` button to trust the notebook.

# #### Global `shap.force_plot`

# To show a global force plot:
# ```python
# ## Fore plot
# shap.force_plot(explainer.expected_value[1], shap_values[:,:,1], features=X_shap)
# 
# 
# ```

# #### Global Force Plot

# In[43]:


## TESTING COMPLEX SHAP VALS AGAIN (Overall Forceplot)
shap.force_plot(explainer.expected_value[1], shap_values[:,:,1].values,features=X_shap)


# #### Fore Plot Interpretation

# - TO DO

# In[ ]:





# In[44]:


# ## Using explainer.shap_values for easier use of force plot
# shap_vals_simple = explainer.shap_values(X_shap)#,y_test)
# print(type(shap_vals_simple))
# shap_vals_simple[0].shape


# In[45]:


# ## Overall Forceplot
# shap.force_plot(explainer.expected_value[1], shap_vals_simple[1],features=X_shap)


# #### Explain Individual Plot

# - To show an individual data point's prediction and the factors pushing it towards one class or another.
# - For now, we will randomly select a row to display, but we will revisit thoughtful selection of examples for stakeholders in our next lesson about local explanations.
# ```python
# ## Just using np to randomly select a row
# row = np.random.choice(range(len(X_shap)))         
# shap.force_plot(explainer.expected_value[1], shap_values[1][row], X_shap.iloc[row])
# ```

# In[46]:


row = np.random.choice(range(len(X_shap)))
print(f"- Row #: {row}")
print(f"- Target: {y_shap.iloc[row]}")
X_shap.iloc[row].round(2)


# In[47]:


# shap_vals_simple[1][row]


# In[48]:


## Individual forceplot (with the complex shap vals)
shap.force_plot(explainer.expected_value[1],shap_values= shap_values[row,:,1].values,
               features=X_shap.iloc[row])


# In[49]:


# ## Individual forceplot
# shap.force_plot(explainer.expected_value[1],shap_values= shap_vals_simple[1][row],
#                features=X_shap.iloc[row])


# ### TEST: (move to next lesson)

# In[50]:


from lime.lime_tabular import LimeTabularExplainer
lime_explainer =LimeTabularExplainer(
    training_data=np.array(X_shap),
    feature_names=X_shap.columns,
    class_names=['Died', 'Survived'],
    mode='classification'
)

exp = lime_explainer.explain_instance(X_shap.iloc[row], rf_clf.predict_proba)
exp.show_in_notebook(show_table=True)


# ### Waterfall Plot

# In[51]:


explainer.expected_value


# In[52]:


shap_values[row,:,1]


# In[53]:


#source: https://towardsdatascience.com/explainable-ai-xai-a-guide-to-7-packages-in-python-to-explain-your-models-932967f0634b
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], 
                                       shap_values[row,:,1].values,
                                       features=X_shap.iloc[row],
                                       show=True)


# ### Interaction Values

# "*The main effects are similar to the SHAP values you would get for a linear model, and the interaction effects captures all the higher-order interactions are divide them up among the pairwise interaction terms. Note that the sum of the entire interaction matrix is the difference between the model’s current output and expected output, and so the interaction effects on the off-diagonal are split in half (since there are two of each). When plotting interaction effects the SHAP package automatically multiplies the off-diagonal values by two to get the full interaction effect.*"
# - https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/NHANES%20I%20Survival%20Model.html#Compute-SHAP-Interaction-Values 

# - Interactions: - https://towardsdatascience.com/analysing-interactions-with-shap-8c4a2bc11c2a

# In[54]:


shap_interaction_values = explainer.shap_interaction_values(X_shap)
shap.summary_plot(shap_interaction_values[0],X_shap)


# In[55]:


shap.dependence_plot(
    ("age", "sex_M"),
    shap_interaction_values[1], X_shap,
    display_features=X_shap
)


# In[56]:


shap.dependence_plot(
    ("goout", "Walc"),
    shap_interaction_values[1], X_shap,
    display_features=X_shap
)


# >- **The more the student goes out, the higher the Walc, and ...(a negative shap interaction value would mean....🤔) `BOOKMARK`**

# > TO DO: read more about the interactions and add interpretation here
# 

# ## `Shap Decision Plot?`
# - https://slundberg.github.io/shap/notebooks/plots/decision_plot.html

# In[75]:


X_shap.loc[( X_shap['sex_M']==1) & (X_shap['Medu']>3) & (X_shap['goout']>2)\
          & (X_shap['reason_course']==1)]


# In[71]:


X_shap['goout']


# In[76]:


example = 13


# In[77]:


shap.decision_plot(explainer.expected_value[1], shap_values[:,:,1].values,X_shap,
                  highlight=example)


# ## 📌 TO DO

# - Try more targets. 
#     - Combine D and F into 1 group
#     - Make a target about decrease in performance from g1 to g3.

# # APPENDIX

# In[58]:


raise Exception('Do not include below in run all.')


# ## Lesson Creation Code

# In[ ]:


# [o for o in dir(shap) if 'Explainer' in o]


# In[ ]:


import pandas as pd
tables = pd.read_html("https://shap.readthedocs.io/en/latest/api.html")
len(tables)


# In[ ]:


explainers = tables[1]#.style.hide('index')
explainers.columns = ['Explainer','Description']
explainers['Explainer'] = explainers['Explainer'].map(lambda x: x.split('(')[0])
explainers


# In[ ]:


print(explainers.set_index("Explainer").to_markdown())


# In[ ]:




