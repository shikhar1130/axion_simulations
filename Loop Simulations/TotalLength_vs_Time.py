import numpy as np
import os
import matplotlib.pyplot as plt


input_directory = r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 1\expanding_universe_testing"
total_sum = []

for i in range(400, 1175, 25):
    
    input_file = os.path.join(input_directory, f"length_{i}txt.npy")
    length = np.load(input_file)
    total_sum.append(np.sum(length))

total_sum = np.array(total_sum)
    

# Plotting
plt.plot(range(400, 1175, 25), total_sum, label="$256^3$")
plt.xlabel("Time steps", fontsize=17)
plt.ylabel("Total Length of Strings", fontsize=17)
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=14)  # Replace 14 with 12
plt.yticks(fontsize=14)  # Replace 14 with 12
ax = plt.gca()
ax.minorticks_on()
plt.legend(fontsize=14)
ax.tick_params(which='minor', top=True, right=True)
ax.tick_params(which='major', top=True, right=True)
ax.tick_params(which='major', length=14, width=1, direction='in')
ax.tick_params(which='minor', length=6, width=0.7, direction='in')
plt.show()