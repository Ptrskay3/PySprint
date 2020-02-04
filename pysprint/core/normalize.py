# This code is by no means part of the main PySprint module. It needs to be embedded and
# should be cleaned up to be dry and robust.

from matplotlib import pyplot as plt
import numpy as np
from pysprint.utils import calc_envelope
from pysprint.core.peak import SelectButton
import matplotlib


class DraggableLine:
    
    # TODO: this should be pixel distance, because x and y are often not on the same scale
    epsilon = 100 # max absolute distance to count as a hit

    def __init__(self, x, y, mode='l'):
        matplotlib.rcParams["toolbar"] = "toolmanager"
        self.fig, self.ax = plt.subplots()
        self.x = x
        self.y = y
        self.mode = mode
        if self.mode == 'l':
            self.lower, self.y_env, self.loc = calc_envelope(
                self.y, np.arange(len(self.y)), 'l'
                )
            plt.title('Adjust the lower envelope.')
        elif self.mode == 'u':
            self.lower, self.y_env, self.loc = calc_envelope(
                self.y, np.arange(len(self.y)), 'u'
                )
            plt.title('Adjust the upper envelope.')
        else:
            raise ValueError('mode must be u or l.')
        self._ind = None # the active point index
        self.basedata, = self.ax.plot(self.x, self.y)
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.lines, = self.ax.plot(self.x, self.lower, 'r')
        self.x_env = self.x[self.loc]
        self.peakplot, = self.ax.plot(
            self.x_env, self.y_env, 'ko'
            )
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_callback)
        self.fig.canvas.mpl_connect('draw_event', self.draw_callback)
        self.fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        tm = self.fig.canvas.manager.toolmanager
        tm.add_tool('Toggle recording', SelectButton)
        self.fig.canvas.manager.toolbar.add_tool(
            tm.get_tool('Toggle recording'), "toolgroup"
            )
        self.my_select_button = tm.get_tool('Toggle recording')
        plt.grid()
        plt.show()


    def button_release_callback(self, event):
        '''whenever a mouse button is released'''
        if event.button != 1:
            return
        self._ind = None


    def get_ind_under_point(self, event):
        '''Get the index of the selected point within the given epsilon tolerance.'''
        d = np.hypot(self.x_env - event.xdata, self.y_env - event.ydata)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None
        return ind


    def button_press_callback(self, event):
        '''whenever a mouse button is pressed we get the index of '''
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)


    def key_press_callback(self, event):
        '''whenever a key is pressed'''
        if not event.inaxes:
            return
        if self.my_select_button.toggled:
            if event.key == 'd':
                ind = self.get_ind_under_point(event)
                if ind is not None:
                    self.x_env = np.delete(self.x_env,
                                             ind)
                    self.y_env = np.delete(self.y_env, ind)
                    idx = np.argsort(self.x_env)
                    self.y_env, self.x_env = self.y_env[idx], self.x_env[idx]
                    self.lower = np.interp(self.x, self.x_env, self.y_env)

                    self.peakplot.set_data(self.x_env, self.y_env)
                    self.lines.set_data(self.x, self.lower)
            elif event.key == 'i':
                self.y_env = np.append(self.y_env, event.ydata)
                self.x_env = np.append(self.x_env, event.xdata)
                idx = np.argsort(self.x_env)
                self.y_env, self.x_env = self.y_env[idx], self.x_env[idx]
                self.lower = np.interp(self.x, self.x_env, self.y_env)
                self.peakplot.set_data(self.x_env, self.y_env)
                self.lines.set_data(self.x, self.lower)
            if self.peakplot.stale:
                self.fig.canvas.draw_idle()

    def get_data(self):
        if self.mode == 'l':
            return self.y-self.lower
        elif self.mode == 'u':
            return self.y/self.lower

    def draw_callback(self, event):
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.peakplot)
        self.ax.draw_artist(self.lines)

    # TODO: We don't really need this, might be removed
    def motion_notify_callback(self, event):
        '''on mouse movement we move the selected point'''
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if self.my_select_button.toggled:
            x, y = event.xdata, event.ydata
            self.x_env[self._ind], self.y_env[self._ind] = x, y
            idx = np.argsort(self.x_env)
            self.y_env, self.x_env = self.y_env[idx], self.x_env[idx]
            self.lower = np.interp(self.x, self.x_env, self.y_env)
            self.peakplot.set_data(self.x_env, self.y_env)
            self.lines.set_data(self.x, self.lower)

            # TODO: make faster drawing with background

            # self.fig.canvas.restore_region(self.background)
            self.ax.draw_artist(self.lines)
            self.ax.draw_artist(self.peakplot)
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.draw()



if __name__ == '__main__':

    import pysprint
    a = pysprint.CosFitMethod.parse_raw('D:/Python/mrs/URES0078.trt')
    a.chdomain()
    # print(a)
    a.slice(2,4)
    # a.guess_GD(-304)
    # a.optimizer(2.355, initial_region_ratio=0.05)

    d = DraggableLine(a.x, a.y, 'l')
    yy = d.get_data()
    d2 = DraggableLine(a.x, yy, 'u')
    yyy = d2.get_data()
    # plt.plot(a.x, yyy)
    # plt.grid()
    # plt.title('Final render')
    # plt.show()
    b = pysprint.CosFitMethod(a.x, yyy)
    # b.smart_guess()
    b.guess_GD(-304)
    b.optimizer(2.355, initial_region_ratio=0.05)
    ## GD = 505.68835 fs^1
    ## GDD = -168.84100 fs^2
    ## TOD = -113.34504 fs^3
    ## with r^2 = 0.98010.

    ## raw norm:
    ## GD = 505.66241 fs^1
    ## GDD = -168.60948 fs^2
    ## TOD = -113.78394 fs^3
    ## with r^2 = 0.94167.

    #
