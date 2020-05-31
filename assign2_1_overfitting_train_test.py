import numpy as np 
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd 

#training process
#x values
x = np.linspace(0, 3, 200)
#y values
y = np.sin(2*math.pi*x)
# random noise gaussian 
noise = np.random.normal(0,0.64,200)
# data with noise/ training data 
data = y + noise
# degree from 0 to 9 
degree = np.linspace(0, 25, 26)
degree = degree.astype(int)
print(degree)
#errors = np.array([])

df = pd.DataFrame(columns=['data','x'])
df['x'] = x
df['data'] = data 

rmse = np.array([])
for item in degree:
   weights = np.polyfit(x, data, item)
   model = np.poly1d(weights)
   predict = model(x)
   MSE = np.square(np.subtract(data,predict)).mean() 
   rmse = np.insert(rmse, len(rmse), math.sqrt(MSE))
   
# testing process 
x_test = np.linspace(4, 5, 100)
#y values
y_test = np.sin(2*math.pi*x_test)
# random noise gaussian 
noise_test = np.random.normal(0,0.64,100)
# data with noise/ training data 
data_test = y_test + noise_test
df_test = pd.DataFrame(columns=['data_test','x_test'])
print(len(x_test))
print(len(data_test))
print(len(noise_test))
df_test['x_test'] = x_test
df_test['data_test'] = data_test 

rmse_test = np.array([])
for item in degree:
   weights = np.polyfit(x_test, data_test, item)
   model = np.poly1d(weights)
   predict = model(x_test)
   MSE = np.square(np.subtract(data_test,predict)).mean() 
   rmse_test = np.insert(rmse_test, len(rmse_test), math.sqrt(MSE))



plt.plot(degree, rmse, label='Training', color='blue')
plt.plot(degree, rmse_test, label='Testing', color='red')
plt.xlabel("M")
plt.ylabel("Erms")
xi = list(range(len(degree)))
plt.xticks(xi, degree)
plt.legend() 
plt.show()