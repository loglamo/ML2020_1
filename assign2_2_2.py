import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Mean and Cov
mean_class_1 = [-2, 1]
cov_class_1 = [[1, 0.8],[0.8, 2]] 
mean_class_2 = [3, -2]
cov_class_2 = [[3, 0.8],[0.8, 2]]
# Generate class1 and class 2
x_1, y_1 = np.random.multivariate_normal(mean_class_1, cov_class_1, 100).T 
x_2, y_2 = np.random.multivariate_normal(mean_class_2, cov_class_2, 100).T 

# create x training dataset from class1 class2 just generated
X_1_train = np.vstack((x_1, y_1))
# create y training dataset of class2
X_2_train = np.vstack((x_2, y_2))
# training data set 
X_train = np.concatenate((X_1_train, X_2_train), 1).T

# create label -1, 1 to class1 and class2
Y_1_train = np.full((1,100), -1)
Y_2_train = np.full((1,100), 1)
Y_train = np.concatenate((Y_1_train, Y_2_train), 1).T


# LinearDiscriminant with Least Square Error
clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
pred = clf.fit(X_train, Y_train) # fit model
w = pred.coef_ # take Weights from model 

# Plot results
x = np.linspace(-5,5,100)
y = -(w[0,0]/w[0,1])*x # Discriminent function 
line = 'y = -' + '(' + str(w[0,0]) + '/' + str(w[0,1]) + ')' + '*' +'x' # string of function

plt.plot(x, y, '-r', label=line) # Plot discriminent function 
plt.plot(x_1, y_1, 'o', label='Class1', color='blue') # Plot class1
plt.plot(x_2, y_2, 'o', label='Class2', color='red') # Plot class2
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.show()


