![Screenshot (14)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/1e61e557-eb30-43e5-954b-771b72142d45)# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

Developed by: Tharun K
RegisterNumber:  212222040172

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="darkseagreen")
plt.plot(x_train,regressor.predict(x_train),color="plum")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="darkblue")
plt.plot(x_test,regressor.predict(x_test),color="plum")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
![Screenshot (10)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/85cf076c-0f01-4726-a972-eaab7975f185)
![Screenshot (11)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/1ad69f90-1f36-4f3d-82bc-0a4ac575c274)
![Screenshot (12)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/c0d07888-2ac2-4b4d-b35e-2956d15671ce)
![Screenshot (13)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/a56b0e76-831a-47a6-bbb8-ad54c6c0b8f4)
![Screenshot (14)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/b18f266a-2e16-4319-b211-685f0591cf54)
![Screenshot (15)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/95a9f47c-f560-4c28-abff-41116d965d0c)
![Screenshot (16)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/d40c9d16-7da1-4205-bb9d-f5ccc058e337)
![Screenshot (17)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/b39bc8a8-5740-48a9-97d9-88c2a4559132)
![Screenshot (18)](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/c84bb5f4-9e1a-4491-b89f-a9779a356efb)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
