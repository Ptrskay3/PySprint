import numpy as np
import matplotlib.pyplot as plt


referencex , referencey=np.loadtxt('reference.txt', unpack = True, delimiter = ',')
samplex, sampley = np.loadtxt('sample.txt', unpack = True, delimiter=',')
initx, inity = np.loadtxt('initSpectrum.txt', unpack = True, delimiter = ',')

y = (inity-referencey-sampley)/(2*np.sqrt(referencey*sampley))
plt.figure()
plt.plot(initx,y)
plt.grid()
plt.show()



