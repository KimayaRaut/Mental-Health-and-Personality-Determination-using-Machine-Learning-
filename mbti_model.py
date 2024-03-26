# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import counter_vector
import pickle

# Importing the dataset mbti
dataset = pd.read_csv('dataset/mbti_1.csv')
x1 = dataset.iloc[:,-1].values
y1 = dataset.iloc[:,0].values

# Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y1 = le.fit_transform(y1)

#preprocessing
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,8675):
    review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[:,-1][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

#Bag of words for mbti
x1 = counter_vector.cv.fit_transform(corpus).toarray()
print(x1)

#train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.2, random_state = 1)

#model
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

# from sklearn.svm import SVC
# classifier = SVC(kernel = 'linear', random_state = 0)
# classifier.fit(X_train, y_train)

#accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

#Making Pickel
# pickle.dump(classifier, open('model1.pkl','wb'))
pickle.dump(classifier, open("pima.pickle.dat", "wb"))
pickle.dump(counter_vector.cv, open("vectorizer.pickle", "wb")) 
pickle.dump(le, open("encoder.pickle", "wb")) 