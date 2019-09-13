"""
Sample generator
"""
import numpy as np 	
# from datetime import datetime

C_LIGHT = 299.793 #nm/fs

def _ensure_input(start, stop, center, resolution):
	if start >= stop:
		raise ValueError('start value must be less than stop')
	if center < start or center > stop:
		raise ValueError('center must be between start and stop')	
	if resolution > (stop-start):
		raise ValueError('resolution is too big')
	else:
		pass

def _disp(x ,GD=0, GDD=0, TOD=0, FOD=0, QOD=0):
	return x*GD+(GDD/2)*x**2+(TOD/6)*x**3+(FOD/24)*x**4+(QOD/120)*x**5


def generatorFreq(start, stop, center ,delay, GD=0, GDD=0, TOD=0, FOD=0, QOD=0, resolution=0.1,
				  delimiter=',',pulseWidth=0.02, includeArms=False):
	_ensure_input(start, stop, center, resolution)
	deltaL = delay 
	omega0 = center 
	window = (8*np.log(2))/(pulseWidth**2)
	lamend = (2*np.pi*C_LIGHT)/start
	lamstart = (2*np.pi*C_LIGHT)/stop
	# stepAmount = (lamend-lamstart+resolution)/resolution
	# lam = np.linspace(lamstart, lamend+resolution,stepAmount)
	lam = np.arange(lamstart, lamend+resolution, resolution) 
	omega = (2*np.pi*C_LIGHT)/lam 
	relom = omega-omega0
	i1 = np.exp(-(relom)**2/(window))
	i2 = np.exp(-(relom)**2/(window))
	i = i1 + i2 + 2*np.cos(_disp(relom, GD=GD, GDD=GDD, TOD=TOD, FOD=FOD, QOD=QOD)+(2*deltaL*omega/C_LIGHT))*np.sqrt(i1*i2) ## ####!!!!!!!!!!!!!!
	if includeArms:
		return omega, i, i1, i2
		# np.savetxt('examples/simulated_'+str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))+'_frequency.txt', np.transpose([omega ,i, i1, i2]), 
			# header = 'freq, int, ref, sam', delimiter = delimiter, comments ='')
	else:
		# np.savetxt('examples/simulated_'+str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))+'_frequency.txt', np.transpose([omega ,i,]), 
			# header = 'freq, int', delimiter = delimiter, comments ='')
		return omega, i, np.array([]), np.array([])



#intenzitásarány

def generatorWave(start, stop, center ,delay, GD=0, GDD=0, TOD=0, FOD=0, QOD=0, resolution=0.1, 
				  delimiter=',',pulseWidth=0.02, includeArms=False):
	_ensure_input(start, stop, center, resolution)
	deltaL = delay 
	omega0 = (2*np.pi*C_LIGHT)/center 
	window = (8*np.log(2))/(pulseWidth**2) # ÁT KELL ÍRNI AZ INTERFÉSZEN
	# stepAmount = (stop-start+resolution)/resolution
	# lam = np.linspace(start, stop+resolution, stepAmount)
	lam = np.arange(start, stop+resolution, resolution) 
	omega = (2*np.pi*C_LIGHT)/lam
	relom = omega-omega0 
	i1 = np.exp(-(relom)**2/(window))
	i2 = np.exp(-(relom)**2/(window))
	i = i1 + i2 + 2*np.cos(_disp(relom, GD=GD, GDD= GDD, TOD=TOD, FOD=FOD, QOD=QOD)+(2*omega*deltaL/C_LIGHT))*np.sqrt(i1*i2)
	if includeArms:
		return lam, i, i1, i2
		# np.savetxt('examples/simulated_'+str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))+'_wavelength.txt', np.transpose([lam ,i, i1, i2]), 
			# header = 'wavelength, int, ref, sam', delimiter = delimiter, comments ='')
	else:
		return lam, i, np.array([]), np.array([])
		# np.savetxt('examples/simulated_'+str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))+'_wavelength.txt', np.transpose([lam ,i]), 
			# header = 'wavelength, int', delimiter = delimiter, comments ='')

