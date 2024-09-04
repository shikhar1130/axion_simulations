import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy
from scipy.fft import fftn, fftshift
import math
from matplotlib import colors
from scipy.optimize import curve_fit


def linear_func(x, m, c):
    return m * x + c


dt = 0.05

output_directory = r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 3\length_of_loop"

dimensions = np.array([101, 150, 200, 256, 384, 512])
time_stop = np.array([55, 135, 230, 340, 570, 795])

save = 0

plt.figure(figsize=(16,9))
plt.scatter(dimensions, time_stop, s=100) #label = f"${dimension}^3$, $nt_d: {damp}$"

popt, _ = curve_fit(linear_func, dimensions, time_stop)
plt.plot(dimensions, linear_func(dimensions, *popt), label=f'Linear Fit (Gradient: {popt[0]:.2f})', linestyle='--', linewidth=1.7, color="green")

plt.ylim(0, 950)
plt.xlabel("$L$", fontsize=17)
plt.ylabel("Decay Time", fontsize=17)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=14)  # Replace 14 with 12
plt.yticks(fontsize=14)  # Replace 14 with 12
ax = plt.gca()
ax.minorticks_on()
plt.text(0.05, 0.97, 'The University of Manchester', transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left', fontweight='bold', fontname='Arial')
plt.text(0.09, 0.93, 'x-axes cut for loop decay', transform=ax.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', fontname='Arial')
plt.legend(fontsize=14)
ax.tick_params(which='minor', top=True, right=True)
ax.tick_params(which='major', top=True, right=True)
ax.tick_params(which='major', length=14, width=1, direction='in')
ax.tick_params(which='minor', length=6, width=0.7, direction='in')
 
if (save == 1):
    plt.savefig(os.path.join(output_directory, "loop_decay_plot.png"), dpi=300)
plt.show()

