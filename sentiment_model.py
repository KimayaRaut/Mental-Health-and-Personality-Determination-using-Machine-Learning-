#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import counter_vector
import pickle


# In[10]:


dataset = pd.read_csv('dataset/Twitter_Sentiment_Analysis.csv')
x2 = dataset.iloc[:,-1].values
y2 = dataset.iloc[:,1].values


# In[11]:


print(x2)


# In[12]:


print(len(x2))


# In[13]:


print(y2)


# In[14]:


print(len(y2))


# In[16]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,31962):
    review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[:,-1][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)


# In[17]:


print(corpus)


# In[19]:

x = counter_vector.cv.fit_transform(corpus).toarray()


# In[20]:


print(x)


# In[21]:


print(len(x))


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size = 0.2, random_state = 1)


# In[23]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)


# In[24]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[26]:


for i in y_pred:
    print(i)


# In[25]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:
pickle.dump(classifier, open("pima.pickle1.dat", "wb"))
pickle.dump(counter_vector.cv, open("vectorizer1.pickle", "wb")) 




