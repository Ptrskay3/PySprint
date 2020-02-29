import pandas as pd

__all__ = ['convert_df']

def convert_df(
	filename, ifg, ref=None, sam=None, delimiter=';', decimal=',',
	skiprows=0, index=False, invert_axis=False, single_axis=False, fmt=None
	):
	"""
	Convert and save given interferogram datafile (either single 
	or along with arms) to PySprint GUI - loadable format.

	Parameters
	----------

	filename: string
	Name of the generated output file.

	ifg: string
	Path to the original interferogram file.

	ref: string, optional
	Path to the reference arm file.

	sam: string, optional
	Path to the sample arm file.

	delimiter: string, optional
	The delimiter in the original interferogram files.
	Default is ';'.

	decimal: string, optional
    Character recognized as decimal separator in the original dataset. 
    Often ',' for European data.
	Default is ','.

	skiprows: int, optional
	Number of rows to skip at the beginning of the file.
	Default is 0.

	index: bool, optional
	Write row numbers (index). Keep this False to make sure it's safely loadable.
	Default is False.

	invert_axis: bool, optional
	Whether to treat first columns as y values in the original interferogram files.
	Default is False.

	fmt: string, optional
	Format string for floating point numbers.
	More info at https://docs.python.org/3.4/library/string.html#format-string-syntax
	
	Returns
	-------
	None
	"""
	ifg_data = pd.read_csv(
		ifg, sep=delimiter, decimal=decimal, skiprows=skiprows, names=['x', 'y']
		)
	if (sam is not None and ref is not None):
		if not single_axis:
			sam_data = pd.read_csv(
				sam, sep=delimiter, decimal=decimal, skiprows=skiprows, names=['x', 'y']
				)
		else:
			sam_data = pd.read_csv(
				sam, sep=delimiter, decimal=decimal, skiprows=skiprows, names=['y']
				)
		samy = sam_data['y'].values if not invert_axis else sam_data['x'].values
		if not single_axis:
			ref_data = pd.read_csv(
				ref, sep=delimiter, decimal=decimal, skiprows=skiprows, names=['x', 'y']
				)
		else:
			ref_data = pd.read_csv(
				ref, sep=delimiter, decimal=decimal, skiprows=skiprows, names=['y']
				)
		refy = ref_data['y'].values if not invert_axis else ref_data['x'].values
	x = ifg_data['x'].values
	y = ifg_data['y'].values
	if invert_axis:
		x, y = y, x
	try:
		fulldf = pd.DataFrame({'x': x, 'y': y, 'ref': refy, 'sam': samy})
	except NameError:
		fulldf = pd.DataFrame({'x': x, 'y': y})
	fulldf.to_csv(filename, sep=',', decimal='.', index=index, float_format=fmt)
	return
