# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: DEVA DHARSHINI.I
RegisterNumber: 212223240026
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## Data.Head():
![Screenshot 2024-10-18 174541](https://github.com/user-attachments/assets/37b240bf-0fcd-48d1-8bee-19b3fcf2dd22)

## Data.info():
![Screenshot 2024-10-18 174550](https://github.com/user-attachments/assets/3ce73bc5-2f5f-4a29-8461-038da612681d)

## isnull() and sum():
![Screenshot 2024-10-18 174605](https://github.com/user-attachments/assets/b30ae367-b3f7-4375-b40e-87d1ddfed10c)

## Data.Head() for salary:
![Screenshot 2024-10-18 174614](https://github.com/user-attachments/assets/f01e5b3a-16d1-48cc-a79d-940b1ef12b7f)

## MSE Value:
![Screenshot 2024-10-18 174623](https://github.com/user-attachments/assets/8d417416-01fd-44c1-8ecb-47e417a1ed66)

## r2 Value:
![Screenshot 2024-10-18 174632](https://github.com/user-attachments/assets/fd006860-95cd-4642-895a-c1414418f346)

## Data Prediction:
![Screenshot 2024-10-18 174649](https://github.com/user-attachments/assets/0db21c6e-0338-4db7-a35d-cc8d9bfcc4a1)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
