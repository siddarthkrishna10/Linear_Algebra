import pandas as pd
import numpy as np

u = np.ones(50)
u_pd = pd.DataFrame(data=u, columns=['u'])
data = pd.read_csv('C:/Users/Siddhardh/Desktop/OiDS Project/Code/LinearAlgebra_Data.csv')
x = data.drop(["D (m)","Driver ID"], axis=1)
x_data = pd.concat([u_pd, x], axis=1)
y_data = data["D (m)"]

from sklearn.linear_model import LinearRegression

model = LinearRegression()
result = model.fit(x_data, y_data)

y_predict = result.predict(x_data)
y_predict = pd.DataFrame(data=y_predict, columns=['y_pred'])

#result.coef_

def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta

theta = np.ones(3)
h = gradientDescent(x_data, y_data, theta, 0.00001, x_data.shape[0], 100)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_data, y_predict))

y_pred_grad = np.dot(x_data, h)
mean_squared_error(y_data, y_pred_grad)
print(mean_squared_error)