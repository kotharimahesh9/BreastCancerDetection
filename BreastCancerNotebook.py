#!/usr/bin/env python
# coding: utf-8

# ## Import the Libraries

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[23]:


dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
dataset.head()


# ## Split the data into training and test sets

# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Build the Classifier 

# In[25]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# ## Predict the Test Set Results

# In[26]:


y_pred = classifier.predict(X_test)


# ## Make the Confusion Matrix and Find Accuracy

# In[27]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# ## k-Fold Cross Validation

# In[28]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# In[ ]:




