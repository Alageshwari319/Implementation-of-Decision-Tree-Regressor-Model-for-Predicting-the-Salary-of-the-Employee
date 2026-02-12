# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ALAGESHWARI V
RegisterNumber:  212224240010  
*/
```

```python
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("drive/MyDrive/ML/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn import metrics
r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

<img width="330" height="234" alt="image" src="https://github.com/user-attachments/assets/50fe7aea-8826-42a3-a122-39ee2415738a" />




<img width="370" height="223" alt="image" src="https://github.com/user-attachments/assets/a8e61093-cd89-4440-8801-bd66e29bbefb" />





<img width="125" height="215" alt="image" src="https://github.com/user-attachments/assets/6abb0248-e359-4fd1-a9df-8f0ab81043e7" />





<img width="266" height="229" alt="image" src="https://github.com/user-attachments/assets/159afc94-5d49-42e3-9cc0-d980217df9fa" />





<img width="108" height="279" alt="image" src="https://github.com/user-attachments/assets/8e46a0b1-19d4-4a5c-8d90-aa2e9890b07e" />





<img width="198" height="40" alt="image" src="https://github.com/user-attachments/assets/77e22708-5ad5-4cac-bb7f-523d6e08c0c1" />





<img width="1674" height="79" alt="image" src="https://github.com/user-attachments/assets/e9539534-d13d-455f-8b16-e23d9eb84b06" />






## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
