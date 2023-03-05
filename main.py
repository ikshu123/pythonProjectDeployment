import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle

df = load_iris()
X=df.data
Y=df.target
dfX=pd.DataFrame(X,columns=['1','2','3','4'])
dfY=pd.DataFrame(Y,columns=['target'])
trainX,testX,trainY,testY=train_test_split(dfX,dfY,test_size=0.2)
model=RandomForestClassifier()
model.fit(trainX,trainY)
print(testX)
pred=model.predict(testX)
print(accuracy_score(testY,pred))
print(confusion_matrix(testY,pred))

# pickle.dump(model, open('model.pkl','wb'))


