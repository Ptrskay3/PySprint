import numpy as np
import matplotlib.pyplot as plt

from pysprint.core.bases.dataset_base import DatasetBase, C_LIGHT
from pysprint.core._generator import generatorFreq, generatorWave

__all__ = ["Generator"]


class Generator(metaclass=DatasetBase):
    """
    Basic dataset generator.
    """

    def __init__(
        self,
        start,
        stop,
        center,
        delay=0,
        GD=0,
        GDD=0,
        TOD=0,
        FOD=0,
        QOD=0,
        resolution=0.1,
        delimiter=",",
        pulse_width=10,
        normalize=False,
        chirp=0,
    ):
        self.start = start
        self.stop = stop
        self.center = center
        self.delay = delay
        self.GD = GD
        self.GDD = GDD
        self.TOD = TOD
        self.FOD = FOD
        self.QOD = QOD
        self.resolution = resolution
        self.delimiter = delimiter
        self.pulse_width = pulse_width
        self.chirp = chirp
        self.normalize = normalize
        self.x = np.array([])
        self.y = np.array([])
        self.ref = np.array([])
        self.sam = np.array([])
        self.plotwidget = plt
        self.is_wave = False

    def __str__(self):
        return self.__repr__()

    # TODO: PEP8 that horrible line below
    def __repr__(self):
        return f"Generator({self.start}, {self.stop}, {self.center}, delay={self.delay}, GD={self.GD}, GDD={self.GDD}, TOD={self.TOD}, FOD={self.FOD}, QOD={self.QOD}, resolution={self.resolution}, delimiter='{self.delimiter}', pulse_width={self.pulse_width}, normalize={self.normalize})"

    def _check_norm(self):
        """
        Do the normalization when we can.
        """
        if len(self.ref) != 0:
            self._y = (self.y - self.ref - self.sam) / (
                2 * np.sqrt(self.sam * self.ref)
            )

    def generate(self, force_wavelength=False):
        """
        Intelligently (kind of) decide what domain we generate the dataset on.
        """
        if force_wavelength:
            self.generate_wave()
        else:
            if self.stop < 100:
                self.generate_freq()
            else:
                self.generate_wave()

    # TODO: Maybe intelligently decide the domain in a new function
    def generate_freq(self):
        """
        Use this to generate the spectrogram in ang. frequency domain.
        """
        self.x, self.y, self.ref, self.sam = generatorFreq(
            self.start,
            self.stop,
            self.center,
            self.delay,
            self.GD,
            self.GDD,
            self.TOD,
            self.FOD,
            self.QOD,
            self.resolution,
            self.delimiter,
            self.pulse_width,
            self.normalize,
            self.chirp,
        )

    def generate_wave(self):
        """
        Use this to generate the spectrogram in wavelength domain.
        """
        self.is_wave = True
        self.x, self.y, self.ref, self.sam = generatorWave(
            self.start,
            self.stop,
            self.center,
            self.delay,
            self.GD,
            self.GDD,
            self.TOD,
            self.FOD,
            self.QOD,
            self.resolution,
            self.delimiter,
            self.pulse_width,
            self.normalize,
            self.chirp,
        )

    def GD_lookup(self):
        return self.GD + self.delay

    def show(self):
        """
        Draws the plot of the generated data.
        """
        self._check_norm()
        if np.iscomplexobj(self.y):
            self.plotwidget.plot(self.x, np.abs(self.y))
        else:
            try:
                self.plotwidget.plot(self.x, self._y, "r")
            except Exception:  # TODO: better exception case
                self.plotwidget.plot(self.x, self.y, "r")
        self.plotwidget.grid()
        self.plotwidget.show()

    # TODO: rewrite this in a more intelligent manner, this is deprecated
    def save(self, name, path=None):
        """
        Saves the generated dataset with numpy.savetxt.

        Parameters:
        ----------

        name: string
            Name of the output file. You shouldn't include the .txt at the end.

        path: string, default is None
            You can also specify the save path.
            e.g path='C:/examplefolder'
            """
        if not name.endswith(".txt"):
            name += ".txt"
        if path is None:
            np.savetxt(
                f"{name}.txt",
                np.transpose([self.x, self.y, self.ref, self.sam]),
                delimiter=self.delimiter,
            )
            print(f"Successfully saved as {name}.")
        else:
            np.savetxt(
                f"{path}/{name}.txt",
                np.transpose([self.x, self.y, self.ref, self.sam]),
                delimiter=self.delimiter,
            )
            print(f"Successfully saved as {name}.")

    def _phase(self, j):
        if self.is_wave:
            lam = np.arange(self.start, self.stop + self.resolution, self.resolution)
            omega = (2 * np.pi * C_LIGHT) / lam
            omega0 = (2 * np.pi * C_LIGHT) / self.center
            j = omega - omega0
        else:
            lamend = (2 * np.pi * C_LIGHT) / self.start
            lamstart = (2 * np.pi * C_LIGHT) / self.stop
            lam = np.arange(lamstart, lamend + self.resolution, self.resolution)
            omega = (2 * np.pi * C_LIGHT) / lam
            j = omega - self.center
        return (
            j
            + self.delay * j
            + j * self.GD
            + (self.GDD / 2) * j ** 2
            + (self.TOD / 6) * j ** 3
            + (self.FOD / 24) * j ** 4
            + (self.QOD / 120) * j ** 5
        )

    def phase_graph(self):
        """
        Plots the spectrogram along with the spectral phase.
        """
        self._check_norm()
        self.fig, self.ax = self.plotwidget.subplots(2, 1, figsize=(8, 7))
        self.plotwidget.subplots_adjust(top=0.95)
        self.fig.canvas.set_window_title("Spectrum and phase")
        try:
            self.ax[0].plot(self.x, self._y, "r")
        except Exception:  # TODO : handle that too
            self.ax[0].plot(self.x, self.y, "r")
        try:
            self.ax[1].plot(self.x, self._phase(self.x))
        except Exception:
            raise ValueError("The spectrum is not generated yet.")

        self.ax[0].set(xlabel="Frequency/Wavelength", ylabel="Intensity")
        self.ax[1].set(xlabel="Frequency/Wavelength", ylabel=r"$\Phi $[rad]")
        self.ax[0].grid()
        self.ax[1].grid()
        self.plotwidget.show()

    @property
    def data(self):
        """
        Unpacks the generated data.
        If arms are given it returns x, y, reference_y, sample_y
        Else returns x, y
        """
        if len(self.ref) == 0:
            return self.x, self.y
        return self.x, self.y, self.ref, self.sam
