import pandas as pd
import os
import nltk
import nltk
import openpyxl
import re
import numpy as np
from nltk import metrics
from nltk.metrics import *
from sklearn import metrics, tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from stop_words import get_stop_words
import openpyxl

os.chdir('C:/Users/kavis/OneDrive/Desktop/Data Mining/Project')
file = pd.read_excel(r'Trending_Videos_CA_Final_Translated_List.xlsx')
file['Translated_Description'] = file['Translated_Description'].replace('&quot;','"',regex=True)
file.to_excel(r'Trending_Videos_CA_Final_Translation_Replacing.xlsx',index=False)
file = pd.read_excel(r'Trending_Videos_CA_Final_Translation_Replacing.xlsx')
word_tokens = []
token = ''
# Step1-Word tokenization
for i in range(len(file['Translated_Description'])):
    token = file['Translated_Description'][i]
    # print("token:", token)
    word_tokens.append(nltk.word_tokenize(token))
print("Word tokenization:",word_tokens[0])
# Step2-Lowercasing
# Iterating over list of list to apply lowercase
word_lower = list(map(lambda x: list(map(lambda y: y.lower(), x)), word_tokens))
print("Lowercase:",word_lower[0])

# Step3-Stop words removal
stop_words = stopwords.words('english')
filtered_words = []
for i in word_lower:
    word_tokens = []
    for j in i:
        if j not in stop_words:
            word_tokens.append(j)
    filtered_words.append(word_tokens)
print("Stop words removal:",filtered_words[0])
# step4-Stemming
from nltk.stem import PorterStemmer
porter = PorterStemmer()
stemming = []
#Taking the list output after stopwords removal
for i in filtered_words:
    stem_words = []
    for j in i:
        stem_words.append(porter.stem(j))
    stemming.append(stem_words)#Appending to stem_words list
# print(type(stemming))
print("Stemming:", stemming[1])

# Step5-Removing punctuation
from string import punctuation

punct_removed = []
for i in stemming:
    re_punct = []
    for j in i:
        re_punct.append(re.sub(r'[^\w\s]', "", j))
    punct_removed.append(re_punct)
print("Removing punctuation:",punct_removed[1])
# Step6-Removing escape sequence characters
strip_a = []
for i in punct_removed:
    strip_words = []
    for j in i:
        strip_words.append(j.strip())
    strip_a.append(strip_words)

print("Removing escape sequence chars:",strip_a[1])

# Step7-Lemmatization
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
lemmat = []
for i in strip_a:
    lemmat_in =[]
    for j in i:
        lemmat_in.append(lemmatizer.lemmatize(j))
    lemmat.append(lemmat_in)

# Build classification model
# step1-word count with count vectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

Y = file['category_id']
res = [' '.join(ele) for ele in lemmat]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#Train test split
X_train,X_test,y_train,y_test=train_test_split(res,Y,test_size=0.3)
# create the transform
# step2-word frequencies with Tf-Idf vector
vectorizer = TfidfVectorizer()
vec_fit_train=vectorizer.fit(X_train)
#Extracting feature names
num_features = len(vectorizer.get_feature_names())
features_vec=vectorizer.get_feature_names()
#Fitting the tfidf transform
train_tfidf = vectorizer.transform(X_train)
test_tfidf = vectorizer.transform(X_test)
print(train_tfidf.shape, test_tfidf.shape)

#Feature selection
from sklearn.feature_selection import SelectKBest, chi2
def perform_feature_selection(train_tfidf, y_train, k_val):
    """ This method is used in order to perform a feature selection by selecting
    the best k_val features from X_train. It does so according to the chi2
    criterion. The method prints the chosen features and creates
    a new instance of X_train with only these features and returns it
    """
    print("**********FEATURE SELECTION**********")
    # Create and fit selector
    selector = SelectKBest(chi2, k=k_val)
    selector.fit(train_tfidf, y_train)
    #Get idxs of columns to keep
    idxs_selected = selector.get_support(indices=True)
    print("Total number of best selected features:",len(idxs_selected))
    train_tfidf = SelectKBest(chi2, k=k_val).fit_transform(train_tfidf, y_train)
    #print("Total number of best selected features:", len(train_tfidf))
    return train_tfidf
perform_feature_selection(train_tfidf, y_train, 50000)

#Building the KNN model
knn = KNeighborsClassifier(n_neighbors=200)
model=knn.fit(train_tfidf,y_train)
#Predicting category
predicted=model.predict(test_tfidf)
#Estimating accuracy
print("Accuracy from KNN model:",metrics.accuracy_score(y_test, predicted))
'''
#Finding even more better K values
# try K=1 through K=25 and record testing accuracy
k_range = range(500, 700)
# We can create Python dictionary using [] or dict()
scores = []
error_rate = []
# We use a loop through the range 500 to 700
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_tfidf,y_train)
    y_pred_knn = knn.predict(test_tfidf)
    scores.append(metrics.accuracy_score(y_test, y_pred_knn))
    error_rate.append(np.mean(y_pred_knn != y_test))
print(scores)


#Take the best score and apply it in confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
print("Confusion matrix from KNN",confusion_matrix(y_test, y_pred_knn))

print("Classification report from KNN",classification_report(y_test,y_pred_knn))
import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show(block=True)
plt.interactive(False)
#plot a Line graph of the error rate
plt.figure(figsize=(10,6))
plt.plot(range(500,700),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show(block=True)
plt.interactive(False)

#Use GridsearchCV-parameter tuning
from sklearn.model_selection import GridSearchCV
weight_options = ["uniform", "distance"]#Defining weights
k_range = range(500, 700)#Defining the range
param_grid = dict(n_neighbors = k_range, weights = weight_options)
#Performing grid search for KNN method
grid_model=GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
grid_model.fit(train_tfidf,y_train)
print ("Best score from grid model",grid_model.best_score_)
print ("Best params from grid model",grid_model.best_params_)
print ("Best estimator from grid model",grid_model.best_estimator_)
'''
#Multinomial Naive Bayes
#Used when we have discrete data(In text learning,we have count of each word to predict class or label)
from sklearn.naive_bayes import MultinomialNB
multi=MultinomialNB()
#Fitting the model and predicting it
multi.fit(train_tfidf,y_train)
multi_pred=multi.predict(test_tfidf)
train_score=multi.score(train_tfidf,y_train)
#Score on training data
print("Score on training data using multinomial NB:",train_score)
#score on testing data
test_score=multi.score(test_tfidf,y_test)
print("Score on testing data using multinomial NB:",test_score)
print("Classification report for testing data using Multinomial Naive Bayes:")
print(classification_report(y_test,multi_pred))
print("Accuracy from Multinomial NB:",metrics.accuracy_score(y_test, multi_pred))

#Decision Trees suffer badly in such high dimensional feature spaces.
#Build Decision tree
dt = DecisionTreeClassifier()
dt.fit(train_tfidf, y_train)
dt_pred = dt.predict(test_tfidf)
#Building classification report,confusion matrix and accuracy
print(classification_report(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))
print("Accuracy from Decision Tree:",metrics.accuracy_score(y_test, dt_pred))

#Support Vector Machine
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
svm=SVC(kernel='linear')
#Fitting SVM transform
svm.fit(train_tfidf, y_train)
svm_pred = svm.predict(test_tfidf)
#Building classification report,accuracy and precision
print("Classification report from SVM",classification_report(y_test, svm_pred))
print("Accuracy from Support Vector Machine:",metrics.accuracy_score(y_test, svm_pred))
#Polynomial kernel
#Building SVM with polynomial kernel
svm_poly=SVC(kernel='poly',degree=8)
#Fitting the model
svm_poly.fit(train_tfidf, y_train)
#Prediction
svm_pred_poly = svm.predict(test_tfidf)
#Finding Accuracy
print("Accuracy from SVM-Poly Support Vector Machine:",metrics.accuracy_score(y_test, svm_pred_poly))

#Gaussian kernel
#Implementing SVM using rbf kernel
svm_gauss=SVC(kernel='rbf')
#Fitting the transform
svm_gauss.fit(train_tfidf, y_train)
#Predicting the category
svm_pred_gauss = svm_gauss.predict(test_tfidf)
#Finding accuracy
print("Accuracy from SVM-Gauss Support Vector Machine:",metrics.accuracy_score(y_test, svm_pred_gauss))

#Sigmoid kernel
import matplotlib.pyplot as plt
#Building SVM using sigmoid kernel
svm_sigmoid=SVC(kernel='sigmoid')
#Fitting the transform
svm_sigmoid.fit(train_tfidf, y_train)
#Predicting
svm_pred_sigmoid = svm_sigmoid.predict(test_tfidf)
#Building confusion matrix
cm = confusion_matrix(y_test, svm_pred_sigmoid)
print("Accuracy from SVM-Sigmoid Support Vector Machine:",metrics.accuracy_score(y_test, svm_pred_sigmoid))
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('SVM Sigmoid Kernel Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

#RandomForest
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets
clf.fit(train_tfidf,y_train)
rf_pred=clf.predict(test_tfidf)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy
print("Accuracy from Random Forest Classifier:",metrics.accuracy_score(y_test, rf_pred))

