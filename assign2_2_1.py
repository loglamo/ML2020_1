import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

mean_class_1 = [-2, 1]
cov_class_1 = [[1, 0.8],[0.8, 2]] 
mean_class_2 = [3, -2]
cov_class_2 = [[3, 0.8],[0.8, 2]]

x_1, y_1 = np.random.multivariate_normal(mean_class_1, cov_class_1, 100).T 
plt.plot(x_1, y_1, 'o', color='blue')
x_2, y_2 = np.random.multivariate_normal(mean_class_2, cov_class_2, 100).T 
plt.plot(x_2, y_2, 'o', color='red')
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.show()