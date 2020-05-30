import numpy as np 
import math
import matplotlib.pyplot as plt

x = np.linspace(0, 3, 100)
y = np.sin(2*math.pi*x)
noise = np.random.normal(0,0.64,100)
signal = y + noise

plt.plot(x, signal, 'o', color='blue')
plt.show()