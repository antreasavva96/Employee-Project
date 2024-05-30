#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install statsmodels


# In[2]:


# Import packages

# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle

# For Stats
import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic


# In[3]:


df = pd.read_csv('Employee.csv')
df.head()

#Education    : The educational qualifications of employees.
#Joining Year : The year each employee joined the company, indicating their length of service.
#City         : The location or city where each employee is based or works.
#Payment Tier : Categorization of employees into different salary tiers.
#Age          : The age of each employee, providing demographic insights.
#Gender       : Gender identity of employees, promoting diversity analysis.
#Ever Benched : Indicates if an employee has ever been temporarily without assigned work.
#Experience   : The number of years of experience employees have in their current field.
#Leave or Not : Target column.


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


df = df.rename(columns={'JoiningYear': 'Joining_Year',
                          'PaymentTier': 'Payment_Tier',
                          'EverBenched': 'Ever_Benched',
                          'ExperienceInCurrentDomain': 'Experience',
                          'LeaveOrNot': 'Left'})

# Display all column names after the update
df.columns


# In[9]:


df.isna().sum()


# In[10]:


df.duplicated().sum()


# 1,889 rows contain duplicates. That is 41% of the data.

# In[11]:


df[df.duplicated()].head()


# In[12]:


# Drop duplicates and save resulting dataframe in a new variable as needed
df1 = df.drop_duplicates(keep='first')

# Display first few rows of new dataframe as needed
df1.head()


# In[13]:


# Convert 'Payment_Tier' to a categorical data type
df_copy = df1.copy()

# Convert 'Ever_Benched' to binary values
df_copy.loc[:, 'Ever_Benched'] = np.where(df_copy['Ever_Benched'] == 'Yes', 1, 0)

# Convert 'Gender' to binary values
df_copy.loc[:, 'Gender'] = np.where(df_copy['Gender'] == 'Male', 1, 0)

# Convert 'Education' to categorical values
df_copy.loc[:, 'Education'] = df_copy['Education'].astype('category').cat.codes


# In[14]:


df_copy.head()


# In[15]:


# Create a boxplot to visualize distribution of `tenure` and detect any outliers
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for Year Experience', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df_copy['Experience'])
plt.show()


# In[16]:


# Get numbers of people who left vs. stayed
print(df_copy['Left'].value_counts())
print()

# Get percentages of people who left vs. stayed
print(df_copy['Left'].value_counts(normalize=True))


# In[17]:


Education_counts = df1['Education'].value_counts()
print(Education_counts)


# In[18]:


plt.figure(figsize=(8, 6))
Education_counts.plot(kind='bar')
plt.title('Distribution of Education')
plt.xlabel('Education')
plt.ylabel('Count')
plt.show()


# In[19]:


# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing `Experience` distributions for `Payment_Tier`, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='Experience', y='Payment_Tier', hue='Left',orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Years of Experience by Payment Tier', fontsize='14')

# Create histogram showing distribution of `Payment_Tier`, comparing employees who stayed versus those who l
sns.histplot(data=df_copy, x='Payment_Tier', hue='Left', multiple='dodge', shrink=0.5, element='bars', ax=ax[1])
ax[1].set_title('Payment Tier histogram', fontsize=14)

# Display the plots
plt.show()


# In[20]:


# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create mosaic plot showing distributions of `Payment_Tier` by Gender
mosaic(
    df1, ['Gender', 'Payment_Tier'],
    label_rotation=45,
    gap=0.01,
    properties={'Gender': {'color': 'red'}, 'Payment_Tier': {'color': 'green'}},
    labelizer=lambda k: f"Payment_Tier {k}",
ax=ax[0])
ax[0].set_title('Mosaic Plot', fontsize=14)

    
# Create histogram showing distribution of `Gender`, comparing employees who stayed versus those who left
sns.histplot(data=df1, x='Gender', hue='Left', multiple='dodge', shrink=0.5, ax=ax[1])
ax[1].set_title('Gender histogram', fontsize='14')
plt.show()


# In[21]:


df1.groupby(['Gender'])['Payment_Tier'].value_counts()


# In[22]:


df1.groupby(['Gender'])['Payment_Tier'].value_counts(normalize=True)


# **Insight:** 80% of men are in the lowest payment tier, compared to only 60% of women, and almost
# 7% of men are in the highest payment tier, compared to 9% of women.
# 
# 

# In[23]:


df1.groupby(['Left'])['Payment_Tier'].value_counts()


# ### Percentage of separation by Payment Tier (2 s.f):
# 
# i. Payment Tier 1: 55% 
# 
# i.e. `77 / 142 * 100 = 55%`
# 
# ii. Payment Tier 2: 151%  
# iii. Payment Tier 3: 51%
# 
# **Insight:** People in Payment Tier 3 have the lowest percentage of leaving while those on tier 2 have the highest.
# 

# In[24]:


df1.groupby(['Left'])['Education'].value_counts()


# **Insights:** 68% of those who left have a bachelor degree.
#               Those who left and have Masters are almost the same in number as those who stayed.
# 

# In[25]:


# Calculate mean and median Experience of employees who left and those who stayed
df1.groupby(['Left'])['Experience'].agg([np.mean,np.median])


# In[26]:


plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df_copy.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':15}, pad=1);


# The correlation heatmap confirms that the payment tier and age and the male gender have some positive correlation with each other, while experience and education have a negative correlation. Finally whether an employee leaves is positively correlated with the joining year, in other words, the more recently an employee has joined the company, the more likely they are to leave.

# ### Modeling Approach A: Logistic Regression Model
# 
# This approach covers implementation of Logistic Regression.

# In[27]:


df1 = pd.get_dummies(df_copy, columns=['Joining_Year', 'City'], prefix=['Joining_Year', 'City'], drop_first=False)


# In[28]:


df1.head() 


# In[29]:


X = df1.drop('Left', axis=1)

# Display the first few rows of the selected features 
X.head()


# In[30]:


y = df1['Left']

y.head() 


# In[31]:


# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


# In[32]:


# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)


# In[33]:


# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)


# In[34]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, 
                                  display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot(values_format='')

# Display plot
plt.show()


# In[35]:


# Create classification report for logistic regression model
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))


# ### Modeling Approach B: Tree-based Model
# This approach covers implementation of Decision Tree and Random Forest. 

# In[36]:


# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# In[37]:


get_ipython().run_cell_magic('time', '', 'tree1.fit(X_train, y_train)')


# In[38]:


# Check best parameters
tree1.best_params_


# In[39]:


# Check best AUC score on CV
tree1.best_score_


# In[40]:


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                        })
  
    return table


# In[41]:


fb=pd.DataFrame(tree1.cv_results_)


# In[42]:


# Get all CV scores
tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
tree1_cv_results


# In[43]:


# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.8],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }  

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
rf = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# In[44]:


get_ipython().run_cell_magic('time', '', 'rf.fit(X_train, y_train)')


# In[45]:


# Define a path to the folder where you want to save the model
path = r'\Users\antre\OneDrive\Υπολογιστής\DataSets D.A'


# In[46]:


def write_pickle(path, model_object, save_as:str):
    '''
    In: 
        path:         path of folder where you want to save the pickle
        model_object: a model you want to pickle
        save_as:      filename for how you want to save the model

    Out: A call to pickle the model in the folder indicated
    '''    

    with open(path + save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)


# In[47]:


def read_pickle(path, saved_model_name:str):
    '''
    In: 
        path:             path to folder where you want to read from
        saved_model_name: filename of pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model


# In[48]:


def write_pickle(path, model_object, save_as:str):
    '''
    In: 
        path:         path of folder where you want to save the pickle
        model_object: a model you want to pickle
        save_as:      filename for how you want to save the model

    Out: A call to pickle the model in the folder indicated
    '''    

    with open(path + save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)


# In[49]:


# Write pickle
write_pickle(path, rf, 'hr_rf')


# In[50]:


rf = read_pickle(path, 'hr_rf')


# In[51]:


# Check best params
rf.best_params_


# In[52]:


# Get all CV scores
rf_cv_results = make_results('random forest cv', rf, 'auc')
print(tree1_cv_results)
print(rf_cv_results)


# In[53]:


def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
  
    return table


# In[54]:


# Get predictions on test data
rf_test_scores = get_scores('random forest1 test', rf, X_test, y_test)
rf_test_scores


# This seems to be a stable, well-performing final model. 
# 
# Plot a confusion matrix to visualize how well it predicts on the test set.

# In[55]:


# Generate array of values for confusion matrix
preds = rf.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf.classes_)
disp.plot(values_format='');


# In[56]:


# Plot the tree
plt.figure(figsize=(85,20))
plot_tree(tree1.best_estimator_, max_depth=6, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()


# In[57]:


tree1_importances = pd.DataFrame(tree1.best_estimator_.feature_importances_, 
                                 columns=['gini_importance'], 
                                 index=X.columns
                                )
tree1_importances = tree1_importances.sort_values(by='gini_importance', ascending=False)

# Only extract the features with importances > 0
tree1_importances = tree1_importances[tree1_importances['gini_importance'] != 0]
tree1_importances


# In[58]:


sns.barplot(data=tree1_importances, x="gini_importance", y=tree1_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()


# 
# The barplot above shows that in this decision tree model, `Joining_Year_2018`, `Payment_Tier`, `Education`, and `City_Pune` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`.

# ### Summary of model results
# 
# **Logistic Regression**
# 
# The logistic regression model achieved precision of 75%, recall of 75%, f1-score of 73% (all weighted averages), and accuracy of 75%, on the test set.
# 
# **Tree-based Machine Learning**
# 
# After conducting feature engineering, the decision tree model achieved AUC of 75.7%, precision of 90.4%, recall of 55.1%, f1-score of 68.5%, and accuracy of 80%, on the test set. The random forest modestly outperformed the decision tree model. 

# ### Conclusion, Recommendations
# 
# The models and the feature importances extracted from the models confirm that employees who **joined** the company in **2018** are most likely to leave. 
# 
# To retain employees, the following recommendations could be presented to the stakeholders:
# 
# * Initiate dialogue with the leader of the group comprised of employees who joined in 2018 to discern their requirements. Subsequently, offer tailored incentives to secure their sustained commitment, thereby preventing attrition. Alternative, encourage the leader to communicate effectively with the remaining team to reinforce a collective commitment and dissuade any inclination toward departure.
# 
# * Give employes bonuses based on their performance, especially to those in the lower payment tier.
# 

# **References:**
# 
# https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset
