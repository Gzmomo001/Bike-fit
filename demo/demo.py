import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(1000)
y = np.random.standard_normal(20)

#plt.plot(y.cumsum())

x = range(len(y))

plt.figure(figsize=(7,4))
#plt.plot(x,y)
plt.plot(y.cumsum(), 'b', lw=1.5)
plt.plot(y.cumsum(), 'ro')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')

# 2
np.random.seed(2000)
y = np.random.standard_normal((20,2)).cumsum(axis=0)
print(y)

y[:,0] = y[:,0]*10

plt.figure(figsize=(7,5))
plt.title('A Simple Plot')
plt.subplot(211)
plt.plot(y[:,0], 'b', lw=1.5, label='1st')
plt.plot(y[:,0], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
#plt.xlabel('index')
plt.ylabel('value')

plt.subplot(212)
plt.plot(y[:,1], 'g', lw=1.5, label='2nd')
plt.plot(y[:,1], 'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
