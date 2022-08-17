import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# import our data
df = pd.read_csv('heart.csv')

print(df.head())

print(df.describe())

# check if there is lack in our data
print(df.isna().sum())

#check if our data 'target' is balanced
print(df['target'].value_counts())
print(df['target'].value_counts().plot(kind='bar'))
print(df.columns)

categorial_val = []       #if the column is smaller/equall than 10
continous_val =[]         #if the column is bigger than 10
for column in df.columns:
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorial_val.append(column)
    else:
        continous_val.append(column)


# plot target vs columns of categorial_val , to conclude about our data
plt.figure(figsize=(15,15))

for i,column in enumerate(categorial_val,1):
    plt.subplot(3,3,i)
    df[df['target']==0][column].hist(bins=20,color='blue',label='have heart disease = no')
    df[df['target'] == 1][column].hist(bins=20,color='red',label='have heart disease = yes')
    plt.legend()
    plt.xlabel(column)

# another type
for i,column in enumerate(continous_val,1):
    plt.subplot(3,2,i)
    df[df['target']==0][column].hist(bins=20,color='blue',label='have heart disease = no')
    df[df['target'] == 1][column].hist(bins=20,color='red',label='have heart disease = yes')
    plt.legend()
    plt.xlabel(column)


# heart disease - age and max heart rate
plt.figure(figsize=(10,8))

plt.scatter(df.age[df.target==1], df.thalach[df.target==1] , color ='red')

plt.scatter(df.age[df.target==0], df.thalach[df.target==0], color ='blue')

plt.title("heart disease - age and max heart rate")
plt.xlabel('age')
plt.ylabel('max heart rate')


#correlation matrix - check if ther is acorrelated feature which we can redundant them
corr_matrix = df.corr()
print(corr_matrix)

#heat map of corr matrix
fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(corr_matrix, annot= True ,fmt=".2f")

bottom , top = ax.get_ylim()

#corr visualization of target with each feature
df.drop('target',axis=1).corrwith(df['target']).plot(kind='bar', title='correlation with target')



#start to work about our model
#data processing
categorial_val.remove('target')
data = pd.get_dummies(df, columns = categorial_val)
scaler = StandardScaler()
cloumns_scaling = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[cloumns_scaling] = scaler.fit_transform(data[cloumns_scaling])


#applying logistic regrssion
X = data.drop('target', axis=1)
y = data['target']

#split our data
X_train ,X_test, y_train , y_test = train_test_split(X,y,test_size=0.3 ,random_state=42)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)

y_train_pred = logistic_regression.predict(X_train)

#check about over\under fitting
print('accuracy score training set :' ,accuracy_score(y_train , y_train_pred))
print('classification report training set :' ,classification_report(y_train, y_train_pred))
print('confusion matrix training set :' , confusion_matrix(y_train ,y_train_pred))

#check our model with test set
print('accuracy score test set :', accuracy_score(y_test, logistic_regression.predict(X_test)))

train_score = accuracy_score(y_train,logistic_regression.predict(X_train))
test_score = accuracy_score(y_test,logistic_regression.predict(X_test))

result_df = pd.DataFrame(data=[["Logistic Regression", train_score , test_score]],
                         columns=['Model', "training accuracy","test accuracy"])
print(result_df)
