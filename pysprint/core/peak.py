import warnings

warnings.filterwarnings("ignore", category=UserWarning) # for matplotlib

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.backend_bases import MouseButton
from matplotlib.backend_tools import ToolToggleBase
from pysprint.utils import get_closest 
# TODO: implement real euclidean metrics with epsilon distance


class SelectButton(ToolToggleBase):
    """
    Toggle button on matplotlib toolbar.
    """
    description = 'Enable click records'
    default_toggled = True
    default_keymap = 't'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EditPeak(object):
	"""
	This class helps to record and delete peaks from a dataset.
	Right clicks will delete the closest (distance is measured with regards to x)
	extremal point found on the graph, left clicks will add a new point.
	Edits can be saved by just closing the matplotlib window.
	Returns the x coordinates of the selected points.
	Note that this class shouldn't be explicitly called by the user.
	"""
	def __init__(self, x, y, x_extremal=None, y_extremal=None):

		# This is here because we do not want other figures
		# to be affected by this.
		matplotlib.rcParams["toolbar"] = "toolmanager"
		
		self.figure = plt.figure()
		self.cid = None
		self.x = x
		self.y = y
		plt.plot(self.x, self.y, 'r')
		self.x_extremal = x_extremal
		self.y_extremal = y_extremal
		if not len(self.x_extremal) == len(self.y_extremal):
			raise ValueError('Data shapes are different')
		self.press()
		self.lins, = plt.plot(
			self.x_extremal, self.y_extremal, 'ko', markersize=6, zorder=99
			)
		plt.grid(alpha=0.7)
		# adding the button to navigation toolbar
		tm = self.figure.canvas.manager.toolmanager
		tm.add_tool('Toggle recording', SelectButton)
		self.figure.canvas.manager.toolbar.add_tool(
			tm.get_tool('Toggle recording'), "toolgroup"
			)
		self.my_select_button = tm.get_tool('Toggle recording')
		plt.show()

	def on_clicked(self, event):
	        """ Function to record and discard points on plot."""
	        ix, iy = event.xdata, event.ydata
	        if self.my_select_button.toggled:
		        if event.button is MouseButton.RIGHT:
		        	ix, iy, idx = get_closest(ix, self.x_extremal, self.y_extremal)
		        	self.x_extremal = np.delete(self.x_extremal, idx)
		        	self.y_extremal = np.delete(self.y_extremal, idx)
		        elif event.button is MouseButton.LEFT:
		        	ix, iy, idx = get_closest(ix, self.x, self.y)
		        	self.x_extremal = np.append(self.x_extremal, ix)
		        	self.y_extremal = np.append(self.y_extremal, iy)
		       	else:
		       		pass
		        self.lins.set_data(self.x_extremal, self.y_extremal)
		        plt.draw()
		        return

	def press(self):
	        """Usual function to connect matplotlib.."""
	        self.cid = self.figure.canvas.mpl_connect(
	        	'button_press_event', self.on_clicked
	        	)

	def release(self):
	        """ On release functionality. It's never called but we will
	        need this later on.."""
	        self.figure.canvas.mpl_disconnect(self.cid)

	@property
	def get_dat(self):
		""" Returns the the selected points."""
		return self.lins.get_data()
