# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Lakshman
RegisterNumber:  212222240001
*/
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) # remove specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.2,random_state= 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear") # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:
### Placement Data:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/6194fe2b-ae65-493e-b580-11b2636608be)

### After Removing Column:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/3d5577cc-bccd-45e0-ae2e-53e0c211148e)

### Checking the null function():
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/6268497a-b1b1-4f4d-9636-8d0840c17602)

### Data duplicates:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/de46665b-8aa5-4757-93b9-598252858b6c)

### Print Data:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/70618bd8-cfc5-42ac-8822-953802a85842)

### X :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/0760b35e-c5c6-4884-9869-4ec8288c268c)

### Y :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/11d28bd0-bde0-4388-94c7-ed2567a1159d)

### Y_Prediction Array :
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/087f0df1-8fc2-4dc9-9b0a-65f301619847)

### Accuracy Value:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/1f0276d8-ebf6-47d4-b1bd-8f4059f78bf6)

### Confusion Matrix;
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/1a6517ea-de45-426c-b272-c4b0e7488ccf)

### Classification Report:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/f4c8ac22-c73d-4226-ae3a-eaad90608d92)

### Prediction of LR:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707265/23adee98-e2a9-4cd4-bb87-9bd46d881e04)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
