from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

boston = load_boston()

data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = pd.Series(data=boston.target, index=data.index)

X = data.drop('MEDV', axis = 1)
Y = data['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 5)

rr = Ridge(alpha=1) #a higher value of alpha restricts the coefficients further
rr.fit(X_train,Y_train)
Y_pred_train = rr.predict(X_train) #predictions on training data
Y_pred = rr.predict(X_test) #predictions on testing data
# We plot predicted Y (y-axis) against actual Y (x-axis). Perfect predictions will lie on the diagonal. We see the diagonal trend, suggesting a 'good' fit

plt.figure(figsize=(5, 4))
plt.scatter(Y_test,Y_pred)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel("Preço Atual - $Y_i$ ")
plt.ylabel("Predição do Preço - $\hat{Y}_i$")
plt.title("Preço atual e Predição - $Y_i$ e $\hat{Y}_i$")
plt.show()

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test,Y_pred)
print('Mean Squared Error: ',mse)
rsq = r2_score(Y_train,Y_pred_train)
print('R-square, Training: ',rsq)
rsq = r2_score(Y_test,Y_pred)
print('R-square, Testing: ',rsq)

#Let's get the coefficients
print('Intercept: ',rr.intercept_) # This gives us the intercept term
print('Coefficients: \n',rr.coef_) # This gives us the coefficients (in the case of this model, just one coefficient)
