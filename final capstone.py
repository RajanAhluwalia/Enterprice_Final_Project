# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import random
import pandas as pd
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template, make_response
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

path = "/home/rajan/Documents/Ai in enterprice/Final_Project/SpamFilterMachineLearning-master/data/"

train_spam_files = [i for i in os.listdir(path+"spam-train")]
train_nonspam_files = [file for file in os.listdir(path+"nonspam-train")]
test_spam_files = [i for i in os.listdir(path+"spam-test")]
train_nonspam_files = [i for i in os.listdir(path+"nonspam-train")]

spam_train, nonspam_train, nonspam_test, spam_test = [], [], [], []

for pfile in train_spam_files :
    with open(path+"spam-train/"+pfile, encoding="latin1") as f:
        spam_train.append(f.read())
for nfile in train_nonspam_files:
    with open(path+"nonspam-train/"+nfile, encoding="latin1") as f:
        nonspam_train.append(f.read())
        
for pfile in test_spam_files :
    with open(path+"spam-test/"+pfile, encoding="latin1") as f:
        spam_test.append(f.read())
for nfile in train_nonspam_files:
    with open(path+"nonspam-train/"+nfile, encoding="latin1") as f:
        nonspam_test.append(f.read())

emails_train = pd.concat([
    pd.DataFrame({"review":spam_train, "label":1}),
    pd.DataFrame({"review":nonspam_train, "label":0}),
], ignore_index=True).sample(frac=1, random_state=1)
    
emails_test = pd.concat([
    pd.DataFrame({"review":spam_test, "label":1}),
    pd.DataFrame({"review":nonspam_test, "label":0}),
], ignore_index=True).sample(frac=1, random_state=1)
    
total_df = pd.concat([emails_train, emails_test])

from variables import contractions
emails_train['review'] = emails_train['review'].str.lower()
emails_test['review'] = emails_test['review'].str.lower()
for w in contractions:
    emails_train['review'] = emails_train['review'].str.replace(w, contractions[w])    
    emails_test['review'] = emails_test['review'].str.replace(w, contractions[w])
    
emails_train['review'] = emails_train['review'].str.replace('[^A-Za-z\\s+]', '')
emails_test['review'] = emails_test['review'].str.replace('[^A-Za-z\\s+]', '')


emails_train['review'] = emails_train.apply(lambda row: word_tokenize(row['review']), axis=1)
emails_test['review'] = emails_test.apply(lambda row: word_tokenize(row['review']), axis=1)


stop = stopwords.words('english')

emails_train['review'] = emails_train['review'].apply(lambda x: [item for item in x if item not in stop])
emails_test['review'] = emails_test['review'].apply(lambda x: [item for item in x if item not in stop])

'''from nltk.stem import PorterStemmer
ps = PorterStemmer()    
emails_train['review'] =emails_train['review'].apply(lambda row: [ps.stem(item) for item in row])
''' 
from nltk.stem import WordNetLemmatizer
lem =  WordNetLemmatizer()   
emails_train['review'] =emails_train['review'].apply(lambda row: [lem.lemmatize(item,pos ='v') for item in row])
emails_test['review'] = emails_test['review'].apply(lambda row: [lem.lemmatize(item,pos ='v') for item in row])

from gensim.models import Word2Vec
model_train_word2vec = Word2Vec(emails_train['review'], min_count=1)

def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
  
  
    return np.asarray(sent_vec) / numw
  
X_train = []
for sentence in emails_train['review']:
    X_train.append(sent_vectorizer(sentence, model_train_word2vec))

X_test = []
for sentence in emails_test['review']:
    X_test.append(sent_vectorizer(sentence, model_train_word2vec))

'''
X_train = pd.DataFrame(X_train, columns = ['reviews_vec'])
X_test = pd.DataFrame(X_test, columns = ['reviews_vec'])
'''

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

Y_train = emails_train.iloc[:,1].to_frame()
Y_test = emails_test.iloc[:,1].to_frame()

from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, Y_train)

y_pred_train = lr.predict(X_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)

accuracy_score(Y_train, y_pred_train)

from sklearn.svm import LinearSVC
final_svm = LinearSVC(C=0.01)
final_svm.fit(X_train, Y_train)

svm_pred = final_svm.predict(X_test)

svm_pred_train = final_svm.predict(X_train)
accuracy_score(Y_test, svm_pred)

accuracy_score(Y_train, svm_pred_train)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Embedding, LSTM, Dropout, Activation, Dropout
from tensorflow.keras.regularizers import l2

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(100,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy", metrics=['accuracy'])

history = model.fit(X_train,Y_train,epochs=20,verbose=1, batch_size = 64, validation_data = (X_test,Y_test),)

model.predict(X_test)

print(X_test.head())

def check_spam(sent):
    global lr
    sent = [sent]
    for w in contractions:
        sent = [item.replace(w, contractions[w]) for item in sent]
    sent = [item.replace('[^A-Za-z\\s+]', '') for item in sent]
    sent = word_tokenize(sent[0])
    sent = [item for item in sent if item not in stop]
    sent = [lem.lemmatize(item,pos ='v') for item in sent]
    #model = Word2Vec(sent, min_count=1)
    sent = sent_vectorizer(sent, model_train_word2vec)
    print(sent)
    def check(y_pred):
        if(y_pred==1):
            return "It is a spam"
        else:
            return "It is not a spam"
        
    print("Using Logistic Regression")
    print(lr)
    y_pred = lr.predict([sent])
    return (check(y_pred))
    #print("Using SVC")
    #y_pred = final_svm.predict([sent])
    #print(check(y_pred))
    
@app.route('/index', methods=['GET', 'POST'])
#app.secret_key="hello rajan"
def index():
    if request.method == 'POST':
        email = request.form['msg']
        
        #email = "0"          
        #email = "Congratulation,you won a lottery, Click here to claim"
        email=str.lower(email)
        flash(check_spam(email))
        #return redirect(url_for('index'))
        
    return render_template('index.html')


if __name__ == "__main__":
    app.secret_key="hello rajan"
    app.run()