from sklearn.model_selection import train_test_split
import dvc.api
import pandas as pd
from sklearn import model_selection, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,ConfusionMatrixDisplay
#from sklearn.externals import joblib
import pickle
import json

with dvc.api.open(repo="https://github.com/devpratap11/MLOps_Assignment", path="data/creditcard.csv", mode="r") as fd:
		df = pd.read_csv(fd)

print('done')

X_train,X_test=train_test_split(df,test_size=0.2)
Y_train=X_train["Class"]
Y_test=X_test["Class"]

train=X_train.to_csv('train.csv')
test=X_test.to_csv('test.csv')

print('Split saved')

X_train=X_train.drop(['Class'],axis=1)
X_test=X_test.drop(['Class'],axis=1)

classifier = RandomForestClassifier()
classifier.fit(X_train,Y_train)
print('model trained')

filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
print('model saved')

#Evaluating the model on test data
Y_pred = classifier.predict(X_test)
a1=accuracy_score(Y_test,Y_pred)
f1=f1_score(Y_test,Y_pred,average='macro')
print("Accuracy: ", a1)
print("Macro F1 score: ", f1)

with open("acc_f1.json", "w") as outfile: 
    json.dump(a1, outfile) 
    json.dump(f1, outfile)  