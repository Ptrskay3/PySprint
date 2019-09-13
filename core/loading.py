'''
This file helps to load in data with auto detection features. Bugs might be encountered, currently working on fix.

'''

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from heapq import nlargest as _nlargest

toLookFor = ['PHz', 'frequency','omega', 'wavelength', 'nm','lambda', 'intensity', 'int', 'ref', 'reference', 'sam', 'sample']

xAxis = toLookFor[:6]
yAxis = toLookFor[6:8]
refAxis = toLookFor[8:10]
samAxis = toLookFor[10:]


def get_matches(word, possibilities, n=3, cutoff=0.7):
    if not n >  0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result = []
    s = SequenceMatcher()
    s.set_seq2(word)
    for idx, x in enumerate(possibilities):
        s.set_seq1(x)
        if s.real_quick_ratio() >= cutoff and \
           s.quick_ratio() >= cutoff and \
           s.ratio() >= cutoff:
            result.append((s.ratio(), idx))
    result = _nlargest(n, result)
    return [x for score, x in result]

def has_header(file, nrows=20):
	try:
		df = pd.read_csv(file, header=None, nrows=nrows)
	except:
		pass
	df_header = pd.read_csv(file, nrows=nrows)
	return tuple(df.dtypes) != tuple(df_header.dtypes)


def read_data(file):
	matches = []
	initSpectrumX = np.array([])
	initSpectrumY = np.array([])
	ref = np.array([])
	sam = np.array([])
	df = pd.read_csv(file)
	if len(df.columns) == 2:
		test = False
	else:
		test = True
	# print(has_header(file))
	if has_header(file):
		for string in toLookFor:
			indexes = get_matches(string, df.columns.to_numpy(), n=1, cutoff = 0.66)
			if len(indexes)>0:
				matches.append([indexes,string])
	else:
		if len(df.columns) == 2:
			df = pd.read_csv(file, header = None, names = ['x','y'])
			initSpectrumX = df.x.values
			initSpectrumY = df.y.values
		elif len(df.columns) == 3:
			df = pd.read_csv(file, header = None, names = ['x','y','z'])
			df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
		elif len(df.columns) == 4:
			df = pd.read_csv(file, header = None, names = ['x','y', 'refY', 'samY'])
			initSpectrumX = df.x.values
			initSpectrumY = df.y.values
			sam = df.samY.values
			ref = df.refY.values
		else:
			df = pd.read_csv(file, header = None)
			df.drop(df.columns[2:], axis = 1, inplace=True)
			initSpectrumX = df.x.values
			initSpectrumY = df.y.values

	df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce', axis=1)
	df = df.dropna(how = 'any', axis = 0)
	# print(matches)
	# print(df)
	for item in matches:
		if item[1] in xAxis:
			try:
				initSpectrumX = df.iloc[:,item[0][0]].values
				df.iloc[0,item[0][0]] = 'crowded'
				if df.iloc[0,item[0][0]] == 'crowded':
					df = df.drop(df.columns[item[0][0]], axis=1)
					# print('df')
			except:
				pass
		elif item[1] in yAxis:
			try:
				initSpectrumY = df.iloc[:,item[0][0]].values
				df.iloc[0,item[0][0]] = 'crowded'
				if df.iloc[0,item[0][0]] == 'crowded':
					df = df.drop(df.columns[item[0][0]], axis=1)
					# print('f')
				# print(initSpectrumY[:3])
			except:
				pass
		elif item[1] in refAxis:
			try:
				ref = df.iloc[:,item[0][0]].values
				df.iloc[0,item[0][0]] = 'crowded'
				if df.iloc[0,item[0][0]] == 'crowded':
					df = df.drop(df.columns[item[0][0]], axis=1)
					# print('ddf')
			except:
				pass

		elif item[1] in samAxis:
			try:
				sam = df.iloc[:,item[0][0]].values
				df.iloc[0,item[0][0]] = 'crowded'
				if df.iloc[0,item[0][0]] == 'crowded':
					df = df.drop(df.columns[item[0][0]], axis=1)
					# print('dfdd')
			except:
				pass

	if len(initSpectrumX) == 0:
		try:
			initSpectrumX = df.iloc[:,0].values
			df = df.drop(df.columns[0], axis=1)
		except:
			pass
	if len(initSpectrumY) == 0:
		try:
			initSpectrumY = df.iloc[:,0].values
			df = df.drop(df.columns[0], axis=1)
		except:
			pass
	if len(ref) == 0:
		if test:
			try:
				ref = df.iloc[:,0].values
				df = df.drop(df.columns[0], axis=1)
				
			except:
				pass
		else:
			pass
	if len(sam) == 0:
		if test:
			try:
				sam = df.iloc[:,0].values
				df = df.drop(df.columns[0], axis=1)
			except:
				pass
		else:
			pass

	return initSpectrumX, initSpectrumY, ref, sam


# x,y,reference,sample = read_data('examples/KURVA.txt')

# print(x[:3])
# print(y[:3])
# print(reference[:3])
# print(sample[:3])
