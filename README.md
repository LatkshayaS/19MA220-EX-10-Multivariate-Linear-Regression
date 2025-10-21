# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
###Step 1:
Import the required modules — matplotlib.pyplot for plotting, numpy for numerical operations, 
and sklearn libraries for datasets, linear regression, and model evaluation.
###Step 2:
Load the Boston housing dataset using datasets.load_boston().
###Step 3:
Define the feature matrix (X) and the response vector (y) from the dataset.
###Step 4:
Split the data into training and testing sets using train_test_split(), 
specifying the test size and random state for reproducibility.
###Step 5:
Create a LinearRegression object using linear_model.LinearRegression()
###Step 6:
Train the regression model using the training data with the fit() function.
###Step 7:
Display the regression coefficients using reg.coef_.
###Step 8:
Calculate and display the variance score (R² value) using reg.score() to evaluate model performance.
###3Step 9:
Set the plot style using plt.style.use() for better visualization.
###Step 10:
(Next step would include plotting residual errors for training and test sets.)
###Step 11:
End the program.


## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

# load the boston dataset
boston = datasets.load_boston(return_X_y=False)

# defining feature matrix (X) and response vector(y)
X = boston.data
y = boston.target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error
plt.style.use('fivethirtyeight')

# plotting residual errors in





```
## Output:
![EX10.png]()

### Insert your output

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
