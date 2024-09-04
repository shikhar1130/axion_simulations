import os
import imageio
import numpy as np

dimension = 101

start = 0
stop = 18
inte = 1

output_directory = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 11\neumann_ellipse\101_v_2R\out"

time_steps = np.arange(start, stop, inte)

images = []
for k in time_steps:
    file_path = os.path.join(output_directory, "101_v_2R_time_" + str(k) + ".png")
    try:
        image = imageio.imread(file_path)
        images.append(image)
        print(k)
    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping...")

gif_path = os.path.join(output_directory, f"{dimension}_v_tilted_R.gif")
imageio.mimsave(gif_path, images, duration=0.1)
del images
