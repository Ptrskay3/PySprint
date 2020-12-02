import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------MONKEY PATCH-------------------------------------
import matplotlib.cbook as cbook
import matplotlib.widgets as widgets
from matplotlib.backend_tools import ToolBase  # lgtm [py/unused-import]
from matplotlib.backend_managers import ToolManager  # lgtm [py/unused-import]


def tm_init(self, figure=None):
    self._key_press_handler_id = None

    self._tools = {}
    self._keys = {}
    self._toggled = {}
    self._callbacks = cbook.CallbackRegistry()

    # to process keypress event
    self.keypresslock = widgets.LockDraw()
    self.messagelock = widgets.LockDraw()

    self._figure = None
    self.set_figure(figure)


matplotlib.backend_managers.ToolManager.__init__ = tm_init


def tb_init(self, toolmanager, name):
    self._name = name
    self._toolmanager = toolmanager
    self._figure = None


matplotlib.backend_tools.ToolBase.__init__ = tb_init

# ----------------------------------------------------------------------------

from matplotlib.backend_tools import ToolToggleBase
from pysprint.utils import _get_closest


class SelectButton(ToolToggleBase):
    """
    Toggle button on matplotlib toolbar.
    """

    description = "Enable click records"
    default_toggled = True
    default_keymap = "t"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EditPeak:
    """
    This class helps to record and delete peaks from a dataset.
    Right clicks will delete the closest extremal point found
    on the graph, left clicks will add a new point.
    Edits can be saved by just closing the matplotlib window.
    Returns the x coordinates of the selected points.
    Note that this class shouldn't be explicitly called by the user.
    """
    def __init__(self, x, y, x_extremal=None, y_extremal=None):

        # This is here because we do not want other figures
        # to be affected by this.
        plt.ion()
        matplotlib.rcParams["toolbar"] = "toolmanager"

        self.figure, self.ax = plt.subplots()
        self.cid = None
        self.x = x
        self.y = y
        plt.plot(self.x, self.y, "r")
        self.x_extremal = x_extremal
        self.y_extremal = y_extremal
        self.press()
        if self.x_extremal is not None and self.y_extremal is not None:
            if not len(self.x_extremal) == len(self.y_extremal):
                raise ValueError("Data shapes are different")
        else:
            self.x_extremal, self.y_extremal = [], []
        (self.lins,) = self.ax.plot(
            self.x_extremal, self.y_extremal, "ko", markersize=8, zorder=99, alpha=0.5
        )
        self.ax.grid(alpha=0.7)
        # adding the button to navigation toolbar
        tm = self.figure.canvas.manager.toolmanager
        tm.add_tool("Toggle recording", SelectButton)
        self.figure.canvas.manager.toolbar.add_tool(
            tm.get_tool("Toggle recording"), "toolgroup"
        )
        self.my_select_button = tm.get_tool("Toggle recording")
        plt.sca(self.ax)
        plt.show(block=True)

    def on_clicked(self, event):
        """
        Function to record and discard points on plot.
        This function is kind of expensive, but we must work in pixel
        coordinates, because otherwise when the axes are differently scaled the
        result will be catastrophicly unintuitive to work with.
        """
        ix, iy = event.xdata, event.ydata

        # change the cursor's position to pixels
        xy_pixels = self.ax.transData.transform(np.vstack([ix, iy]).T)
        ix, iy = xy_pixels.T

        # compute the current pixel positions of the curve
        xy_pixels = self.ax.transData.transform(np.vstack([self.x, self.y]).T)
        x_pix, y_pix = xy_pixels.T

        if event.inaxes is None:
            return
        if self.my_select_button.toggled:
            if event.key == "d":
                # compute extremals' pixel coords
                extr_pixels = self.ax.transData.transform(np.vstack([self.x_extremal, self.y_extremal]).T)
                extr_x, extr_y = extr_pixels.T

                # get the closest extremal to the cursor in pixels
                ix, iy, idx = _get_closest(ix, iy, extr_x, extr_y)

                # delete the corresponding index in original array
                self.x_extremal = np.delete(self.x_extremal, idx)
                self.y_extremal = np.delete(self.y_extremal, idx)

            elif event.key == "i":
                # compute the closest point to the cursor in pixels
                ix, iy, idx = _get_closest(ix, iy, x_pix, y_pix)

                # change back to data coords and append
                pixels = self.ax.transData.inverted().transform(np.vstack([ix, iy]).T)
                x_data, y_data = pixels.T

                self.x_extremal = np.append(self.x_extremal, x_data)
                self.y_extremal = np.append(self.y_extremal, y_data)

            self.lins.set_data(self.x_extremal, self.y_extremal)
            plt.draw()
            return

    def press(self):
        """Usual function to connect matplotlib.."""
        self.cid = self.figure.canvas.mpl_connect("key_press_event", self.on_clicked)

    def release(self):
        """ On release functionality. It's never called but we will
        need this later on.."""
        self.figure.canvas.mpl_disconnect(self.cid)

    @property
    def get_dat(self):
        """ Returns the the selected points."""
        return self.lins.get_data()
