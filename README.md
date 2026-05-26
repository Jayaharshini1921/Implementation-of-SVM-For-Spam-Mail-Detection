# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 Detect File Encoding: Use chardet to determine the dataset's encoding.
2 Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3 Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4 Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5 Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6 Train SVM Model: Fit an SVC model on the training data.
7 Predict Labels: Predict test labels using the trained SVM model.
8 Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Jayaharshini s
RegisterNumber: 212224100024
*/
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
<img width="708" height="150" alt="398595784-ba42ca2d-85a5-4795-b40d-63df37236a81" src="https://github.com/user-attachments/assets/22a593eb-d46a-47cc-a5fd-d8a8855b6b40" />
<img width="726" height="237" alt="398595804-22edcf48-dd25-4513-867d-6c7430da985d" src="https://github.com/user-attachments/assets/bc14771d-abc4-4a8e-8d78-84e658d4a3da" />
<img width="414" height="274" alt="398595822-618324db-35cd-4c45-8469-466a22c5cf86" src="https://github.com/user-attachments/assets/4712806b-4250-4c1b-a305-c8d2a8acc280" />
<img width="220" height="169" alt="398595845-81bc5619-eb01-41f1-9218-e95a64875d42" src="https://github.com/user-attachments/assets/8908eeaf-26fa-4796-88db-59e0715dc8ed" />
<img width="653" height="184" alt="398595874-6836f245-d141-4e8b-8bfa-e2d8eac18a76" src="https://github.com/user-attachments/assets/b70acad9-999c-49d1-a451-d9bf7cee5905" />
<img width="422" height="131" alt="398595912-3da73419-4d11-45bf-b361-b1fd344e732e" src="https://github.com/user-attachments/assets/a32bdc8b-2066-4fbb-a348-e1f72191c6d3" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
