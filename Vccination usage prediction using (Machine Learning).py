#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[3]:


data = pd.read_csv('h1n1_vaccine_prediction.csv')
data.dtypes


# In[4]:


data = data.drop(columns=['unique_id'])
data.columns


# In[5]:


data_category_column = data.columns[data.dtypes=='object']  #Which of the columns are category type


# In[6]:


data_numeric_column = data.columns[data.dtypes!='object'] #Which of the columns are of numeric type


# In[7]:


print(data_category_column) #Names of category columns


# In[8]:


print(data_numeric_column) #Names of numeric columns


# In[10]:


#Create data sub sets
data_category = data[data_category_column]  #Category Data
data_numeric = data[data_numeric_column]   #Numeric Data
data_category.shape
data_numeric.shape
data_category.head()
data_numeric.head()


# In[11]:


#Check for missing data
data_category.isna().sum().sort_values(ascending=False)
data_numeric.isna().sum().sort_values(ascending=False)


# In[12]:


#Categorical imputation using lambda functionlambda x: x.fillna(x.value_counts(). index[0]
#For Sample
y = pd.DataFrame(['Male','Male',np.NaN,np.NaN,np.NaN,'Female','Male'])
print(y.value_counts())

y=y.apply(lambda x: x.fillna(x.value_counts().index[0]))

y
data_clean_category = data_category.apply(lambda x:x.fillna(x.value_counts().index[0]))
data_clean_category.isna().sum().sort_values(ascending=False)
data_clean_category.dtypes


# In[13]:





# In[20]:


data_numeric.head()
data_numeric.dtypes
data_clean_numeric = data_numeric.apply(lambda 
x: x.fillna(x.value_counts().index[0]))
data_clean_numeric.head()
data_clean_numeric.isna().sum()
data_clean_numeric.dtypes
data_clean_numeric = data_clean_numeric.astype('category')
data_clean_numeric.head()
data_clean_numeric['h1n1_worry'].value_counts()
category_count = data_clean_category.columns.value_counts()

sns.barplot(category_count.index,category_count.values,alpha=0.9)

plt.show()
numeric_count = data_clean_numeric.columns.value_counts()
sns.barplot(numeric_count.index,numeric_count.values,alpha=0.9)

plt.show()
data_clean_category.columns
data_clean_numeric.columns
data_clean_category.dtypes
data_clean_numeric.dtypes
data_clean_category
data_clean_category['age_bracket'] = data_clean_category['age_bracket'].cat.codes

data_clean_category['qualification'] =data_clean_category['qualification'].cat.codes

data_clean_category['race']=data_clean_category['race'].cat.codes

data_clean_category['sex']=data_clean_category['sex'].cat.codes

data_clean_category['income_level']=data_clean_category['income_level'].cat.codes

data_clean_category['marital_status']=data_clean_category['marital_status'].cat.codes
data_clean_category['housing_status']=data_clean_category['housing_status'].cat.codes
data_clean_category['employment']=data_clean_category['employment'].cat.codes

data_clean_category['census_msa'] = data_clean_category['census_msa'].cat.codes

data_clean_category.head()
data_clean_numeric.isna().sum()
data_clean_numeric = data_clean_numeric.apply(lambda x: x.fillna(x.value_counts().index[0]))
data_clean_numeric.isna().sum()
print(type(data_clean_category))
print(type(data_clean_numeric))


# In[21]:


covid_data = pd.concat([data_clean_category,data_clean_numeric],
                       join='outer',
                      axis=1)

covid_data.head()
covid_data.corr()
X = covid_data.iloc[:,:-1]

y = covid_data.iloc[:,-1]
X.shape,y.shape


# In[22]:


X = pd.get_dummies(X)
X.shape
X.head()
sns.heatmap(X)
covid = pd.concat([X,y],axis=1)
covid.head()
type(covid)


# In[25]:


X = covid.iloc[:,:-1].values  #features

y = covid.iloc[:,-1].values    #target
X,y,X.shape,y.shape
from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)
Xtrain.shape,Xtest.shape,ytrain.shape,ytest.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model
model.fit(Xtrain,ytrain)
ypred=model.predict(Xtest)
ytest
ypred
model.coef_   #Weights or coefficient
model.intercept_

from sklearn import model_selection 
 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
print(sklearn.metrics.accuracy_score(ytest,ypred))
print(sklearn.metrics.classification_report(ytest,ypred))
print(sklearn.metrics.confusion_matrix(ytest,ypred))
(4049+430) / (4049+194+669+430)
Vaccinators = pd.DataFrame(ypred)  #ypred is in array.It needs to be converted to DataFrame

Vaccinators.value_counts()
print('Default Logistic Regression Accuracy Training Score',model.score(Xtrain,ytrain))
print('Default Logistic Regression Accuracy Test Score',model.score(Xtest,ytest))


# In[26]:


X = covid.iloc[:,:-1].values  #features

y = covid.iloc[:,-1].values    #target
X,y,X.shape,y.shape
from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)
Xtrain.shape,Xtest.shape,ytrain.shape,ytest.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model
model.fit(Xtrain,ytrain)
ypred=model.predict(Xtest)
ytest
ypred
model.coef_   #Weights or coefficient
model.intercept_

from sklearn import model_selection 
 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
print(sklearn.metrics.accuracy_score(ytest,ypred))
print(sklearn.metrics.classification_report(ytest,ypred))
print(sklearn.metrics.confusion_matrix(ytest,ypred))
(4049+430) / (4049+194+669+430)
Vaccinators = pd.DataFrame(ypred)  #ypred is in array.It needs to be converted to DataFrame

Vaccinators.value_counts()
print('Default Logistic Regression Accuracy Training Score',model.score(Xtrain,ytrain))
print('Default Logistic Regression Accuracy Test Score',model.score(Xtest,ytest))


# In[31]:


covid.shape  #Checking records and variables
import statsmodels.api as sm
from statsmodels.formula.api import logit  #We want to use Logistic Regression in StatsModels
covid_float = covid.astype('float')  #statsmodels require dataset to have float values
x =covid_float.iloc[:,:-1]

y = covid_float.iloc[:,-1]
mle_model = logit("y~X",covid_float,).fit()   #Equation of MLE using statsmodels
print(mle_model.summary())
print(mle_model.summary2())
mle_model.pred_table()    #Build Confusion Matrix
accuracy = (19942+2398)/(19942+1091+3276+2398)
accuracy
mle_model.aic


# In[41]:


from sklearn.linear_model import SGDClassifier
sgd_classifier = SGDClassifier()
sgd_classifier
SGDClassifier(def_init__(loss='hinge', penalty='12', alpha=0.0001, l1_ratio=0.15, 
fit_intercept=True,  max_iter=1000, tol=0.001, shuffle=True, verbose=0, 
epsilon=DEFAULT_EPSILON, n_jobs=None, random_state=None, 
learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False))
sgd_classifier = SGDClassifier(loss='hinge',
                               penalty='l2',
                               max_iter=1000)
sgd_classifier
sgd_classifier.fit(Xtrain,ytrain)
print('Training Accuracy Score',sgd_classifier.score(Xtrain,ytrain))
print('Validation Accuracy Score',sgd_classifier.score(Xtest,ytest))
sgd_classifier.predict(Xtest)
sgd_classifier.coef_
sgd_classifier.intercept_
sgd_classifier.decision_function(Xtest)


# In[42]:


sgd_classifier = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                               early_stopping=False, epsilon=0.1, eta0=0.0,        fit_intercept=True,
                               l1_ratio=0.15, learning_rate='optimal', loss='log',
                               max_iter=5, n_iter_no_change=5, n_jobs=None, penalty='l2',
                               power_t=0.5, random_state=None, shuffle=True, tol=0.001,
                               validation_fraction=0.1, verbose=0, warm_start=False).fit(Xtrain,ytrain)
sgd_classifier.predict_proba(Xtest)
print(sgd_classifier.predict_proba(Xtest))
sgd_ypred = sgd_classifier.predict(Xtest)
sklearn.metrics.accuracy_score(ytest,sgd_ypred)


# In[44]:


#Step 5: Build models 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Spot Check Algorithms 
models = [] 
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC()))
models.append(('DecisionTree',DecisionTreeClassifier()))
models.append(('Random Forest',RandomForestClassifier()))
# evaluate each model in turn 
results = [] 
names = [] 
for name, model in models: 
 kfold = model_selection.KFold(n_splits=10, random_state=None) 
 cv_results = model_selection.cross_val_score(model, Xtrain, ytrain, cv=kfold, scoring='accuracy') 
 results.append(cv_results) 
 names.append(name) 
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
 print(msg)
#Now we can make prediction on most accurate model 
# Make predictions on validation dataset 
rf = RandomForestClassifier() 
rf.fit(Xtrain, ytrain) 
predictions = rf.predict(Xtest) 
print(accuracy_score(ytest, predictions)) 
print(confusion_matrix(ytest, predictions)) 
print(classification_report(ytest, predictions))


# In[ ]:




