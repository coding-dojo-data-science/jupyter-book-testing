# Insights for Stakeholders

## Lesson Objectives
By the end of this lesson, students will be able to:
- Define the stakeholder and their business problem that will be the guiding framework for this week's lessons.
- Identify the target and features for a machine learning model to use for insights
- Understand how different feature/target choices can provide a different spin/perspective on the problem.

## The Stakeholder

- We've been hired by a school district that wants to use data science to identify and support high school students at risk of poor performance. 

- They have provided us with data on several hundred of their former students and want to identify students who are at risk of poor grades/performance by year 3.
    - The school included 3 years of grades(presumable grades 10-12), which are labeled as (G1-G3).
    - They sent us an excel file with 3 sheets:
        - student-mat: grades for Math (the student-mat sheet)
        - student-por: grades for Portuguese
        - README: data dictionary

- The goal is to identify these students and provide additional support/tutoring to improve their academic performance.

### Our Task

- Develop machine-learning models to predict student performance in their final year (year 3).
- We will then use our model(s) to extract insights into which students are most at-risk for poor performance. 
- We will provide a summary of our findings and 3 recommendations to the school on how to best identify these at-risk students. 


We will be focusing on the Math grades to start.

### Stakeholder Considerations

- Before diving into modeling, let's examine the data the school district has provided us and let's brainstorm/discuss our approach for our models.

# EDA 


```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

## Customization Options
# pd.set_option("display.max_columns",100)
plt.style.use(['seaborn-talk'])
mpl.rcParams['figure.facecolor']='white'
```


```python

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS6xDKNpWkBBdhZSqepy48bXo55QnRv1Xy6tXTKYzZLMPjZozMfYhHQjAcC8uj9hQ/pub?output=xlsx"

df = pd.read_excel(url,sheet_name='student-mat')
df.info()
df.head()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 395 entries, 0 to 394
    Data columns (total 33 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   school      395 non-null    object 
     1   sex         395 non-null    object 
     2   age         395 non-null    float64
     3   address     395 non-null    object 
     4   famsize     395 non-null    object 
     5   Pstatus     395 non-null    object 
     6   Medu        395 non-null    float64
     7   Fedu        395 non-null    float64
     8   Mjob        395 non-null    object 
     9   Fjob        395 non-null    object 
     10  reason      395 non-null    object 
     11  guardian    395 non-null    object 
     12  traveltime  395 non-null    float64
     13  studytime   395 non-null    float64
     14  failures    395 non-null    float64
     15  schoolsup   395 non-null    object 
     16  famsup      395 non-null    object 
     17  paid        395 non-null    object 
     18  activities  395 non-null    object 
     19  nursery     395 non-null    object 
     20  higher      395 non-null    object 
     21  internet    395 non-null    object 
     22  romantic    395 non-null    object 
     23  famrel      395 non-null    float64
     24  freetime    395 non-null    float64
     25  goout       395 non-null    float64
     26  Dalc        395 non-null    float64
     27  Walc        395 non-null    float64
     28  health      395 non-null    float64
     29  absences    395 non-null    float64
     30  G1          395 non-null    float64
     31  G2          395 non-null    float64
     32  G3          395 non-null    float64
    dtypes: float64(16), object(17)
    memory usage: 102.0+ KB



    
![png](Insights%20for%20Stakeholders_files/output_11_1.png)
    



```python
# increasing the col width so we can read the description
pd.options.display.max_colwidth = 200
data_dict = pd.read_excel(url,sheet_name='README',usecols=[0,1,2],
                          skiprows=1,index_col=0)
data_dict
```


    
![png](Insights%20for%20Stakeholders_files/output_12_0.png)
    


### Exploratory Data Analysis

- We need to get an overview of this dataset and we'd like to get it quickly. While it is good practice to manually perform the EDA steps, we can also leverage additional tools/packages for quick dataset inspection.


```python
def summarize_df(df_):
    df = df_.copy()
    report = pd.DataFrame({#'# Col':range(len(df.columns)),
#             'Column':df.columns,             
             
            'dtype':df.dtypes,
             '# null': df.isna().sum(),
             'null (%)': df.isna().sum()/len(df)*100,
        'nunique':df.nunique(),
        "min":df.min(),
        'max':df.max()
             })
#     report = report.reset_index(drop=True).set_index("Col #")
    report.index.name='Column'
    return report.reset_index()#.set_index('# Col')

summarize_df(df)
```


    
![png](Insights%20for%20Stakeholders_files/output_15_0.png)
    


## Determining Our Target

- The stakeholder wants to identify students based on their predicted performance in their final year. 
    - "G1","G2","G3" are the student final score at the end of years 1-3.


- One approach we could take is treat this as a regression task, with G3 as our target.
    - This would allow us to predict the exact final grade of each student.
    
    

### `Types<?>` of Model-Based Insights

There are several approaches available to us for modeling-based insights. 

First, we will examine big-picture insights for the entire model/dataset. Some options include:
- Use a Linear Regression to extract coefficients.
    - Unscaled: exact effect on final grade of each feature.
    - Scaled: features that are the most important for final grade.
    
- Use tree-based regression models for Feature Importance.
    - Which features are the most helpful for predicting grade?  (built-in importance)
    - Which features damage the model's performance the most when shuffled? (permutation importance).


We could also treat this as a classification task by creating a "at-risk" or "under-performing" classification column based on grades.
    - We can then leverage additional model explanation tools to understand which features make a student more likely to under-perform.
    
Second, we will select stakeholder-appropriate visualizations to summarize our findings.
- We can leverage additional explanation packages to help us illustrate how a specific student's features influence their predicted performance.  
    


## EDA

- Before we dive into modeling, let's take a moment for some EDA visualizations to help us get a sense of our target. 


### Visualizing Grades


```python
ax = sns.histplot(data=df,x='G3');
ax.set_title("Distribution of Final Grades")
```


    
![png](Insights%20for%20Stakeholders_files/output_23_1.png)
    


- Lets compare all 3 years of grades on one histogram. Since each grade is a separate column, one approach we could take is to use the histplot function 3 times, once for each column.


```python
## Use histplot 3 times
sns.histplot(data=df,x='G3',color='r',alpha=0.9,kde=True,label='G3');
sns.histplot(data=df,x='G2',color='b',alpha=0.7,kde=True,label='G2');
sns.histplot(data=df,x='G1',color='slategray',alpha=0.6,kde=True,label='G1');
plt.title('Compare Years 1-3 (Original Vers)')
plt.legend();
```


    
![png](Insights%20for%20Stakeholders_files/output_25_0.png)
    


- Another way we can leverage seaborn's functionality is to make a new version of our dataframe where the columns "G1"-"G3" will be transposed so that we will have a 1 column with all of the grades and another column with the label for which year the grade was from.


```python
## Let's turn the integer index into a col for student-id
df = df.reset_index()
df = df.rename({'index':'student'},axis=1)
df
```


    
![png](Insights%20for%20Stakeholders_files/output_27_0.png)
    



```python
melted = pd.melt(df,id_vars='student', value_vars=['G1','G2','G3'],var_name='PrevYear',
                value_name="Grade")
melted
```


    
![png](Insights%20for%20Stakeholders_files/output_28_0.png)
    



```python
ax = sns.histplot(data=melted,hue='PrevYear', x='Grade',kde=True)
ax.set_title('Comparing Years 1-3 (Melted Vers)')
```




    Text(0.5, 1.0, 'Comparing Years 1-3 (Melted Vers)')




    
![png](Insights%20for%20Stakeholders_files/output_29_1.png)
    


- Visualizing all 3 grades at the same time, we can see there is actually a gap in the middle of the scores. It seems no student ever receives a grade around 10. This may be something to investigate further in the future.

- It also looks like The # of students with scores near 0 increased each year. 
    - Is there a better way we can see how students performance changed each year?

### Visualizing Progress By Grade Groups

- Let's define groups for each year's scores by binning the scores.




```python
df.describe()[['G1','G2','G3']]
```


    
![png](Insights%20for%20Stakeholders_files/output_33_0.png)
    



```python
## defining bins and their labels
bins = [0,8,14,20]

bin_labels = ['1. low','2. med','3. high']
```


```python
## Visualizing bin cutoffs
ax = sns.histplot(data=melted,hue='PrevYear', x='Grade',kde=True)
ax.set_title('Comparing Years 1-3 (Melted Vers)')
ax.axvspan(bins[0],bins[1],zorder=0,color='lightcoral',label='Low',alpha=0.6)
ax.axvspan(bins[1],bins[2],zorder=0,color='skyblue', label='Med',alpha=0.6)
ax.axvspan(bins[2],bins[3],zorder=0,color='limegreen', label='High',alpha=0.6)
ax.set_xlim(left=0,right=20)
```




    (0.0, 20.0)




    
![png](Insights%20for%20Stakeholders_files/output_35_1.png)
    


- ADD A BRIEF EXPLANATION


```python
pd.cut(df['G1'], bins,right=False,include_lowest=True)#,labels=bin_labels)
```




    0        [0, 8)
    1        [0, 8)
    2        [0, 8)
    3      [14, 20)
    4        [0, 8)
             ...   
    390     [8, 14)
    391    [14, 20)
    392     [8, 14)
    393     [8, 14)
    394     [8, 14)
    Name: G1, Length: 395, dtype: category
    Categories (3, interval[int64, left]): [[0, 8) < [8, 14) < [14, 20)]




```python
bins
```




    [0, 8, 14, 20]




```python
df['First-Year Group'] = pd.cut(df['G1'],bins,labels=bin_labels, right=True,include_lowest=True)
```


```python
df['Final-Year Group'] = pd.cut(df['G3'],bins,labels=bin_labels,right=True,include_lowest=True)
df
```


    
![png](Insights%20for%20Stakeholders_files/output_40_0.png)
    



```python
df[df['Final-Year Group'].isna()]
```


    
![png](Insights%20for%20Stakeholders_files/output_41_0.png)
    



```python
df['Final-Year Group'].value_counts(dropna=False)
```




    2. med     220
    1. low     102
    3. high     73
    Name: Final-Year Group, dtype: int64




```python
melted = pd.melt(df,id_vars=['student','First-Year Group','Final-Year Group'], 
                 value_vars=['G1','G2','G3'],var_name='PrevYear',
                value_name="Grade")
melted
```


    
![png](Insights%20for%20Stakeholders_files/output_43_0.png)
    



```python
bins_info = [('1. low',{ 'bins':(bins[0],bins[1]), 
                    'line_color':'darkred','span_color': 'lightcoral'}),
             ('2. med',{ 'bins':(bins[1],bins[2]),
                    'line_color':'blue','span_color':'skyblue'}),
              ('3. high',{ 'bins':(bins[2],bins[3]),
                      'line_color':'forestgreen','span_color':'limegreen'})
            ]
bins_info
```




    [('1. low',
      {'bins': (0, 8), 'line_color': 'darkred', 'span_color': 'lightcoral'}),
     ('2. med', {'bins': (8, 14), 'line_color': 'blue', 'span_color': 'skyblue'}),
     ('3. high',
      {'bins': (14, 20), 'line_color': 'forestgreen', 'span_color': 'limegreen'})]




```python
## slice out list of line colors
line_colors = [v[1]['line_color'] for v in bins_info]
line_colors
```




    ['darkred', 'blue', 'forestgreen']




```python
## Plot visual using Final year Groups for colors
ax = pd.plotting.parallel_coordinates(df,'Final-Year Group',cols=['G1','G2','G3'],
                                 color= line_colors,sort_labels=False,alpha=0.5)
ax.set_title("Year 1-3 grades - Colored by Final Grade Groups");

## Annotate
for group, grp_dict in bins_info:
    ax.axhspan(grp_dict['bins'][0],grp_dict['bins'][1],
               color=grp_dict['span_color'], alpha=0.5, label=group)

ax.set_ylim(bottom=0,top=20)
```




    (0.0, 20.0)




    
![png](Insights%20for%20Stakeholders_files/output_46_1.png)
    


- Looking at our parallel coordinates plot - grouped by Final Grades, we can see that there are students who were in the low group for G3 that had been in the med-group range. So students whose performance dropped from average to below average over the years.
- We can also see that there were some students in the high group whose G1 scores were in the med range, showing that they improved their performance over years 1-3.
- We also notice that there are a lot of zeroes that appear in G2 and G3, what we can see all that lines that were 0 for G2 stayed at 0 for G3.

## Next Steps
-  Now that we have an idea of the target we are analyzing, we are ready to move onto modeling!

- We will start by exploring answering our stakeholder's question(s) with regression modeling.
- Let's get started!
