#!/usr/bin/env python
# coding: utf-8

# # Predicting heart disease using machine learning
# 
# This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.
# 
# We're going to take the following approach:
# 
# 1.Problem definition
# 2.Data
# 3.Evaluation
# 4.Features
# 5.Modelling
# 6.Experimentation
# 
# 
# # Problem Definition
# In a statement,
# 
# Given clinical parameters about a patient, can we predict whether or not they have heart disease? 
# 
# 
# # DATA
# 
# The original data came from the Cleavland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+Disease
# 
# There is also a version of it available on Kaggle. https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset
# 
# 
# # Evaluation
# If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.
# 
# # Features
# This is where you'll get different information about each of the features in your data. You can do this via doing your own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).
# 
# ## Data Dictionary
# 
# 1. **age**: Age in years.
# 
# 2. **sex**: Gender of the individual. 
#     - 1: Male
#     - 0: Female
# 
# 3. **cp**: Chest pain type.
#     - 0: Typical angina
#     - 1: Atypical angina
#     - 2: Non-anginal pain
#     - 3: Asymptomatic
# 
# 4. **trestbps**: Resting blood pressure (in mm Hg).
#     - Values above 130-140 are typically cause for concern.
# 
# 5. **chol**: Serum cholesterol in mg/dl.
#     - Serum = LDL + HDL + 0.2 * Triglycerides
#     - Values above 200 are cause for concern.
# 
# 6. **fbs**: Fasting blood sugar level > 120 mg/dl.
#     - 1: True
#     - 0: False
#     - '>126' mg/dL signals diabetes.
# 
# 7. **restecg**: Resting electrocardiographic results.
#     - 0: Nothing to note
#     - 1:ST-T Wave abnormality: 
#     Can range from mild symptoms to severe problems, signals non-normal heartbeat.
#     - 2:Possible or definite left ventricular hypertrophy:
#     Enlarged heart's main pumping chamber.
# 
# 8. **thalach**: Maximum heart rate achieved.
# 
# 9. **exang**: Exercise-induced angina.
#     - 1: Yes
#     - 0: No
# 
# 10. **oldpeak**: ST depression induced by exercise relative to rest. Reflects the stress of the heart during exercise. An unhealthy heart will stress more.
# 
# 11. **slope**: The slope of the peak exercise ST segment.
#     - 0:Upsloping: Better heart rate with exercise (uncommon)
#     - 1:Flatsloping: Minimal change (typical healthy heart)
#     - 2:Downslopins: Signs of unhealthy heart
# 
# 12. **ca**: Number of major vessels (0-3) colored by fluoroscopy.
#     - Colored vessel means the doctor can see the blood passing through. 
#     -The more blood movement, the better (no clots).
# 
# 13. **thal**: Thallium stress result.
#     - 1,3: Normal
#     - 6: Fixed defect (used to be defect but okay now)
#     - 7: Reversible defect: No proper blood movement when exercising.
# 
# 14. **target**: Presence of heart disease.
#     - 1: Yes
#     - 0: No
# 
# ## Tools Used
# 
# We're going to use the following tools for data analysis and manipulation:
# - Pandas
# - Matplotlib
# - NumPy
# 

# In[87]:


#IMPORTING ALL TOOLS WE NEED 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay

# LOAD DATA 
# In[14]:


df = pd.read_csv("heart-disease.csv")
df.shape # (rows, columns)


# In[15]:


df.head()


# In[16]:


df.tail()


# In[17]:


#finding out how many of each class are there
df["target"].value_counts()


# In[21]:


#lets visualize this 
df["target"].value_counts().plot(kind="bar", color=["salmon","green"]);


# In[22]:


#info about data
df.info()


# In[24]:


#check for missing values
df.isna().sum()


# In[25]:


#want to know some other parameters 
df.describe()

FINDING PATTERNS IN OUR DATA Heart Disease Frequency according to Sex
# In[26]:


df.sex.value_counts()


# In[27]:


# lets Compare target column with sex column
pd.crosstab(df.target, df.sex)


# In[28]:


#lets visualize
pd.crosstab(df.target, df.sex).plot(kind="bar",
                                    figsize=(10, 6),
                                    color=["salmon", "lightblue"])

plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Diesease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"]);
plt.xticks(rotation=0);

Age vs. Max Heart Rate for Heart Disease
# In[29]:


pd.crosstab(df.age, df.target)


# In[30]:


#lets visualize

plt.figure(figsize=(10, 6))
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="salmon")
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="lightblue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# *here we can observe a pattern as age decreases heart rate decreases
Heart Disease Frequency per Chest Pain Type
3.cp - chest pain type
 0: Typical angina: chest pain related decrease blood supply to the heart
 1: Atypical angina: chest pain not related to heart
 2: Non-anginal pain: typically esophageal spasms (non heart related)
 3: Asymptomatic: chest pain not showing signs of disease

# In[31]:


pd.crosstab(df.cp, df.target)


# In[32]:


#lets visualize

pd.crosstab(df.cp, df.target).plot(kind="bar",
                                   figsize=(10, 6),
                                   color=["salmon", "lightblue"])

plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease", "Disease"])
plt.xticks(rotation=0);


# CORRELATION MATRIX
# 

# In[36]:


df.corr()


# In[37]:


#lets visualize
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="coolwarm");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

MODELLING
# In[38]:


df.head()


# In[39]:


#split your data into training and test sets
X = df.drop("target",axis=1)
y=df["target"]


# In[40]:


X


# In[41]:


y


# In[42]:


#splitting our data into train and test sets
np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)


# In[43]:


X_train


# In[44]:


y_train, len(y_train)


# we got our data split into train and test sets,it's time to build models 
# We'll train it (find the patterns) on the training set.
# 
# And we'll test it (use the patterns) on the test set.
# 
# We're going to try 3 different machine learning models:
# 
# Logistic Regression
# K-Nearest Neighbours Classifier
# Random Forest Classifier

# In[45]:


# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    """
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[46]:


model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)

model_scores

What exactly does the code above does?

So basically what it's going to do is it's going to take our dictionary of models.
It's going to set up a random seed.
It'll set up an empty dictionary.
And it's going to loop through our model's dictionary.
So for name model.
So for logistic regression, logistic regression in models, dot items.
Logistic regression will pretend it's logistic regression dot fit.
So selling the logistic regression model to find the patterns in the training data.
And then it's going to create a key in our empty dictionary model scores with the score of how well
our logistic regression model performs on the test data, a.k.a. using the patterns that it's found in the training data.
And then it's going to return our model scores dictionary, and then we're going to see how each of
our model performs on the test data set and our three models here, we should have three different scores.MODEL COMPARISON
# In[49]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();

Hyperparameter tuning (by hand)
# In[50]:


# Let's tune KNN

train_scores = []
test_scores = []

# Create a list of differnt values for n_neighbors
neighbors = range(1, 21)#default n_neighbours is only 5 
#we are tuning the hyper parameters
# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)#range(1, 21)
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    # Update the training scores list
    train_scores.append(knn.score(X_train, y_train))
    
    # Update the test scores list
    test_scores.append(knn.score(X_test, y_test))


# In[51]:


train_scores


# In[52]:


test_scores

its better visualizing 
# In[53]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# By using hyper parameter tuning we were able to improve the score from 68% to 75% was achived when number of neighbours are increased to 11 from 5

# So instead of us having to manually try tuning different type of parameters by hand randomized search CV, it's going to tRy a number of different combinations of hyper parameters for us and evaluate which ones are the best and then save them for us
Hyperparameter tuning with RandomizedSearchCV
We're going to tune:

LogisticRegression()
RandomForestClassifier()
... using RandomizedSearchCV
# In[54]:


# Create a hyperparameter grid for LogisticRegression and RandomForestClassifier
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# In[57]:


np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)


# In[58]:


#finding best parameters
rs_log_reg.best_params_


# In[59]:


rs_log_reg.score(X_test, y_test)

There is no change in logisticRegression will try with RandomForestClassifier
# In[73]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(), 
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)


# In[75]:


rs_rf.best_params_


# In[76]:


rs_rf.score(X_test, y_test)


# we improved score of our model from 83% to 86%

# Hyperparamter Tuning with GridSearchCV
# Since our LogisticRegression model provides the best scores so far, we'll try and improve them again using GridSearchCV...

# In[77]:


log_reg_grid = {"C": np.logspace(-4, 4, 30),
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)


gs_log_reg.fit(X_train, y_train);


# In[78]:


gs_log_reg.best_params_


# In[79]:


gs_log_reg.score(X_test, y_test)

Evaluting our tuned machine learning classifier, beyond accuracy
 ROC curve and AUC score
 Confusion matrix
 Classification report
 Precision
 Recall
 F1-score
... and it would be great if cross-validation was used where possible.

To make comparisons and evaluate our trained model, first we need to make predictions.
# In[80]:


y_preds = gs_log_reg.predict(X_test)


# In[81]:


y_preds


# In[82]:


y_test


# In[88]:


#plotting ROC curve
y_proba = gs_log_reg.predict_proba(X_test)[:, 1]

roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
roc_display.plot()
plt.show()


# In[94]:


print(confusion_matrix(y_test, y_preds))


# In[95]:


import seaborn as sns
sns.set(font_scale=1.5) # Increase font size
 
def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=False)
    plt.xlabel("Predicted label") # predictions go on the x-axis
    plt.ylabel("True label") # true labels go on the y-axis 
    
plot_conf_mat(y_test, y_preds)

we got a ROC curve, an AUC metric and a confusion matrix, let's get a classification report as well as cross-validated precision, recall and f1-score.
# In[96]:


print(classification_report(y_test, y_preds))

Calculate evaluation metrics using cross-validation
We're going to calculate accuracy, precision, recall and f1-score of our model using cross-validation and to do so we'll be using cross_val_score().
# In[97]:


gs_log_reg.best_params_


# In[98]:


# Create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")


# In[99]:


# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="accuracy")
cv_acc

In cv it evaluates the model in 5 different splits so we can considder mean 
# In[103]:


cv_acc = np.mean(cv_acc)
cv_acc


# In[101]:


# Cross-validated precision
cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")
cv_precision=np.mean(cv_precision)
cv_precision


# In[108]:


#Cross-validated recall
cv_recall = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall


# In[104]:


# Cross-validated f1-score
cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1


# In[109]:


# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "F1": cv_f1},
                          index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                      legend=False);

Feature Importance
Feature importance is another as asking, "which features contributed most to the outcomes of the model and how did they contribute?"

Finding feature importance is different for each machine learning model. One way to find feature importance is to search for "(MODEL NAME) feature importance".

Let's find the feature importance for our LogisticRegression model...
# In[110]:


# Fit an instance of LogisticRegression
#better we use the best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")

clf.fit(X_train, y_train);


# In[111]:


clf.coef_
#checking coefficients 

The array represents how much each attribute contributes will convert into something readable

# In[112]:


feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# LETS VISUALIZE THIS FOR BETTER UNDERSTANDING

# In[113]:


feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);

6. Experimentation
# WE CAN STILL EXPERIMENT WITH THE DATA AND MAKE THE MODEL MORE EFFICIENT 
