# For combined plot of length:

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def linear_func(x, m, c):
    return m * x + c

dimension = [101, 150, 200, 256, 512]

dx = 0.25
dy = dx
dz = dx
dt = dx / 5

damp = 0

start = [35, 65, 100, 140, 345]
end = [55, 135, 230, 340, 795]

start2 = [35, 65, 100, 140, 385]
end2 = [55, 135, 230, 300, 620]

interval = 5

plt.figure(figsize=(16,9))
for k, j in enumerate(dimension):
    output_directory = f"C:\\Users\\shikh\\OneDrive - The University of Manchester\\Docs\\UOM\\UoM\\Course\\Sem8\\MPhys\\MPhys Project\\Week 3\\length_of_loop\\new_ic\\{dimension[k]}\\out"
    total_sum = []  # Reset total_sum for each dimension
    
    for i in range(start[k], end[k], interval):
        input_file = os.path.join(output_directory, f"length_{i}.npy")
        length_calc = np.load(input_file)
        total_sum.append(np.sum(length_calc))
    
    total_sum = np.array(total_sum)
    time_steps = np.arange(start2[k], end2[k], interval)  # Use np.arange for better precision
    
    # Plotting
    plt.scatter(range(start[k], end[k], interval), total_sum, s=7, label=f"${dimension[k]}^3$")
    
    total_sum_interp = interp1d(np.arange(start[k], end[k], interval), total_sum, kind='linear')
        
    popt, _ = curve_fit(linear_func, time_steps, total_sum_interp(time_steps))
    plt.plot(time_steps, linear_func(time_steps, *popt), label=f'Linear Fit (Gradient: {popt[0]:.2f})', linestyle='--', linewidth=1.7)
 
plt.xlabel("Time steps", fontsize=17)
plt.ylabel("Total Length of Strings", fontsize=17)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax = plt.gca()
ax.minorticks_on()
plt.legend(fontsize=14)
plt.text(0.05, 0.97, 'The University of Manchester', transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left', fontweight='bold', fontname='Arial')
plt.text(0.09, 0.93, 'x-axes cut for loop decay', transform=ax.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', fontname='Arial')
ax.tick_params(which='minor', top=True, right=True)
ax.tick_params(which='major', top=True, right=True)
ax.tick_params(which='major', length=14, width=1, direction='in')
ax.tick_params(which='minor', length=6, width=0.7, direction='in')

# plt.show()
plt.savefig("combined_fit.png", dpi=300)
