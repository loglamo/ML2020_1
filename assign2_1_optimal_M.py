import numpy as np 
import math
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import pandas as pd 

#x values
x = np.linspace(0, 3, 100)
#y values
y = np.sin(2*math.pi*x)
# random noise gaussian 
noise = np.random.normal(0,0.64,100)
# data with noise/ training data 
data = y + noise
# degree from 0 to 9 
degree = np.linspace(0, 9, 10)
print(degree)

errors = np.array([])
for item in degree:
    df = pd.DataFrame(columns=['data','x'])
    df['x'] = x
    df['data'] = data 
    weights = np.polyfit(x, data, item)
    model = np.poly1d(weights)
    predict = model(x)
    results = smf.ols(formula='data ~ model(x)', data=df).fit()
    error = results.mse_total
    mean_error = math.sqrt(error)
    errors = np.insert(errors,len(errors), mean_error)
print(errors)
#plt.plot(x, data,'o',color='blue')
#plt.plot(x, y , color='green')
#plt.plot(x, predict, color='red')
#plt.xlabel("x")
#plt.ylabel("t")
#plt.show()

#testing 
#x values
x_test = np.linspace(3, 6, 100)
#y values
y_test = np.sin(2*math.pi*x)
# random noise gaussian 
noise_test = np.random.normal(0,0.64,100)
# data with noise/ training data 
data_test = y_test + noise
# degree from 0 to 9 

errors_test = np.array([])
for item in degree:
    df = pd.DataFrame(columns=['data_test','x_test'])
    df['x_test'] = x_test
    df['data_test'] = data_test 
    weights = np.polyfit(x_test, data_test, item)
    model = np.poly1d(weights)
    predict = model(x_test)
    results = smf.ols(formula='data_test ~ model(x_test)', data=df).fit()
    error = results.mse_total
    mean_error = math.sqrt(error)
    errors_test = np.insert(errors_test,len(errors_test),mean_error)

print(errors_test)
plt.plot(degree, errors,color='red') 
plt.plot(degree, errors_test,color='blue')
plt.show()
