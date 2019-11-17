import pandas as pd
import numpy as np

a = np.ones(50)
a_pd = pd.DataFrame(data=a, columns=['a'])
data = pd.read_csv('LinearAlgebra_Data.csv')
x = data.drop(["D (m)","Driver ID"], axis=1)
x_data = pd.concat([a_pd, x], axis=1)
y_data = data["D (m)"]

from sklearn.linear_model import LinearRegression

model = LinearRegression()
result = model.fit(x_data, y_data)

y_predict = result.predict(x_data)
y_predict = pd.DataFrame(data=y_predict, columns=['y_pred'])

def gradient_Descent(x, y, t, alpha, m, iterations):
    xTrans = x.transpose()
    for i in range(0, iterations):
        hypothesis = np.dot(x, t)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        grad = np.dot(xTrans, loss) / m
        t = t - alpha * grad
    return t

t = np.ones(3)
h = gradient_Descent(x_data, y_data, t, 0.00001, x_data.shape[0], 100)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_data, y_predict))

y_pred_grad = np.dot(x_data, h)
mean_squared_error(y_data, y_pred_grad)
