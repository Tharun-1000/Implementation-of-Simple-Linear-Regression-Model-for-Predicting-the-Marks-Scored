 # Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.


2. Assign hours to X and scores to Y.


3. Implement training set and test set of the dataframe.


4. Plot the required graph both for test data and training data.


5. Find the values of MSE , MAE and RMSE.

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


1.df.head()

![267700887-6917a59e-d5b2-4cb8-ad9d-c89935d0db36](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/08cf3b3f-419b-45ee-a2fb-94142e6e8520)


2.df.tail()

![image](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/d76b3e28-fdc5-4f96-b43b-864da8fdfbd9)


3.Array value of X

![image](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/99ac6f59-81fd-4e1f-9594-0dc15425d1e5)

4.Array value of Y

![image](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/60420830-851f-4767-875b-2265fe50bf58)

5.Values of Y prediction

![image](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/eeed566f-b571-4ec4-80bf-02de230a3018)

6.Array values of Y test

![267701155-819d7d3a-6d5e-4430-84b2-b9962d477946](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/fb6565eb-cc32-4d99-9aaa-0c8470a8606a)

7.Training Set Graph

![image](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/bda02448-20f5-49fa-9312-8dd7eba79e21)

8.Test Set Graph

![image](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/304d438c-7207-433c-9a1b-d3792711d8d1)

9.Values of MSE, MAE and RMSE

![image](https://github.com/Tharun-1000/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/135952958/3fe64691-4fed-474e-86df-3060bce4ffde)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
