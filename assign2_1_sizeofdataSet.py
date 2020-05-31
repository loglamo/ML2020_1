import numpy as np 
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd 

#training process
#x values
x = np.linspace(0, 3, 300)
#y values
y = np.sin(2*math.pi*x)
# random noise gaussian 
noise = np.random.normal(0,0.64,300)
# data with noise/ training data 
data = y + noise
# degree from 0 to 9 
degree = 10
print(degree)
#errors = np.array([])

df = pd.DataFrame(columns=['data','x'])
df['x'] = x
df['data'] = data 
weights = np.polyfit(x, data, degree)
model = np.poly1d(weights)
predict = model(x)

   
# N=50
x_test = np.linspace(0, 3, 30)
#y values
y_test = np.sin(2*math.pi*x_test)
# random noise gaussian 
noise_test = np.random.normal(0,0.64,30)
# data with noise/ training data 
data_test = y_test + noise_test
df_test = pd.DataFrame(columns=['data_test','x_test'])
print(len(x_test))
print(len(data_test))
print(len(noise_test))
df_test['x_test'] = x_test
df_test['data_test'] = data_test 
weights_50 = np.polyfit(x_test, data_test, degree)
model_50 = np.poly1d(weights)
predict_50 = model(x_test)

plt.plot(x_test, predict_50, label='N = 30', color='red')
plt.plot(x_test, data_test,'o', color='blue')
plt.plot(x_test, y_test,color='green')
plt.xlabel("x")
plt.ylabel("t")
plt.legend() 
plt.show()