from pymongo import MongoClient
import pandas as pd
client = MongoClient()

client = MongoClient('localhost', 27017)

db = client.dbBackup

#Loading the dataset
collection1 = db.attempts
attempts = pd.DataFrame(list(collection1.find()))


#removing the dict variables
attempts=attempts.drop('subjects',axis=1)
attempts=attempts.drop('practiceSetInfo',axis=1)
attempts=attempts.drop('createdBy',axis=1)
attempts=attempts.drop('attemptdetails',axis=1)
attempts=attempts.drop('__v',axis=1)
attempts=attempts.drop('_id',axis=1)

attempt.to_csv('attempts.csv')

#seeing the uniwue values first
for column in attempts.columns:
    print(column +" has- "+str(len(attempts[column].unique())))
    
#Seeing the datatype of the columns
attempts.info()    

#Seeoing the missing value
df=attempts
for x in df.columns:
    if df[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,df[x].isnull().values.ravel().sum()))
        
#removing the columns that have no effect on scores

attempts=attempts.drop('attemptType',axis=1)
attempts=attempts.drop('createdAt',axis=1)
attempts=attempts.drop('email',axis=1)
attempts=attempts.drop('idOffline',axis=1)
attempts=attempts.drop('practicesetId',axis=1)
attempts=attempts.drop('studentName',axis=1)
attempts=attempts.drop('updatedAt',axis=1)
attempts=attempts.drop('user',axis=1)
attempts=attempts.drop('userId',axis=1)

df=attempts.copy()

df.info()

#verifying for no missing value
for x in df.columns:
    if df[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,df[x].isnull().values.ravel().sum()))
    else:
        print("no missing value in "+x)
        
        
#Mapping
#boolean type to int
df.isAbandoned = df.isAbandoned.astype(int)
df.isAnsync = df.isAnsync.astype(int)
df.isCratedOffline = df.isCratedOffline.astype(int)
df.isEvaluated = df.isEvaluated.astype(int)
df.isShowAttempt = df.isShowAttempt.astype(int)
df.isfraudelent = df.isfraudelent.astype(int)

df.columns

#Correlation Matrix #conclude that some columns have only one value
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
corr=df.corr()
sns.heatmap(corr, annot=True, fmt=".3f", linewidths=.5)
plt.yticks(rotation=0) 
plt.xticks(rotation=70) 
                
for column in df.columns:
    print(column +" has- "+str(len(df[column].unique())))
    
#removing those columns
df=df.drop('isEvaluated',axis=1)
df=df.drop('isShowAttempt',axis=1)
df=df.drop('isfraudelent',axis=1)
df=df.drop('partial',axis=1)
df=df.drop('pending',axis=1)    

#Now again analyzing the corr mat
plt.figure(figsize=(10,10))
corr=df.corr()
sns.heatmap(corr, annot=True, fmt=".3f", linewidths=.5)
plt.yticks(rotation=0) 
plt.xticks(rotation=70)  

#removing columns based on corr mat
df=df.drop('isAbandoned',axis=1)
df=df.drop('isAnsync',axis=1)
df=df.drop('isCratedOffline',axis=1)


df.columns
#rearranging the columns
df=df[['maximumMarks', 'minusMark', 'offscreenTime', 'plusMark',
       'totalCorrects', 'totalErrors', 'totalMarkeds',
       'totalMissed', 'totalQuestions', 'totalTime', 'totalMark']]

#slitting
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second,third and forth hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

classifier.summary()
# Fitting the ANN to the Training set
history=classifier.fit(X_train, y_train, batch_size = 64,validation_split=0.3,nb_epoch = 50)

result = classifier.evaluate(X_test, y_test)
historydf = pd.DataFrame(history.history, index=history.epoch)

historydf.plot(ylim=(0,1))
plt.title("Test accuracy: {:3.1f} %".format(result[1]*100), fontsize=15)
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=y_pred.round()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.scatter(X_train,y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))