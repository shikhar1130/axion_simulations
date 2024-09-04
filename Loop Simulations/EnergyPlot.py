

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy

dx = 0.25
dy = 0.25
dz = 0.25
dt = 0.05
ofset = 0
dv = dx*dy*dz

nx = 101
ny = nz = nx

start = 0
stop = 500

eps = 10**-6
inte = 10
dir_out = r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 6\loop_fixing\v_0.1R\101\out2"
dir_in = r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 6\loop_fixing\v_0.1R\101\DS"
# dir_in2 = r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 4\loop_in_middle\256\GifData"
pos_max = nx/2

lengthCutOff = (dx**2+dy**2+dz**2)**(0.5)
param = nx*dx*.5
# val = np.linspace(0,2,21)
val = [0.8]
val2 = [0.0035]


pot=1
tot=0

for j in range(start, stop, inte):
 #calculate energy and save it   
 
    input_file = dir_in + f"/Energydensity_Global_static_physical_nx{nx}_dampednt0_ntHeld0_seed22_"+str(j)+".txt"
    data = np.loadtxt(input_file)
    print("Data loaded.")
    energy_pot = 0
    energy_tot = 0

 #transform coordinates and save a np array of energy densities
 
    data_array = np.zeros((nx,ny,nz)) 
    data_array2 = np.zeros((nx,ny,nz))

    
    #----------------------------------------------
    
    # Calculate indices for x, y, and z
    x_indices = ((data[:, 0] + param) / dx).astype(int)
    y_indices = ((data[:, 1] + param) / dy).astype(int)
    z_indices = ((data[:, 2] + param) / dz).astype(int)
    
    # Get values for total energy density
    values_tot = data[:, 4]
    values_pot = data[:, 3]
    
    # Assign values to the data_array using calculated indices
    data_array[x_indices, y_indices, z_indices] = values_tot
    data_array2[x_indices, y_indices, z_indices] = values_pot
    
    print("Processing done (energy).")

    
    if tot==1:
        for p in range(0,len(val)):    
    
            figu = plt.figure(figsize=(10, 9))
            az = figu.add_subplot(111, projection='3d')
    
            X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), -np.arange(nz))
    
            kw = {
                'vmin': data.min(),
                'vmax': data.max(),
                'levels': np.linspace(data_array.min(), data_array.max(), 10),
            }
    
            for i in range(nz):
                az.contour(
                    X[:, :, i], Y[:, :, i], data_array[:, :, i],
                    levels=[val[p]], colors=[plt.cm.jet(p / len(val))], zdir='z', offset=-i,linewidths = 0.2,
                )
    
            for i in range(nx):
                az.contour(
                    X[i, :, :], Y[i, :, :], data_array[i, :, :],
                    levels=[val[p]], colors=[plt.cm.jet(p / len(val))], zdir='y', offset=0,linewidths = 0.2,
                )
    
            az.contour(
                data_array[:, -1, :], Y[:, -1, :], Z[:, -1, :],
                levels=[val[p]], colors=[plt.cm.jet(p / len(val))], zdir='x', offset=nx,linewidths = 0.2,
            )
    
            # Set limits of the plot from coord limits
            xmin, xmax = X.min(), X.max()
            ymin, ymax = Y.min(), Y.max()
            zmin, zmax = Z.min(), Z.max()
            az.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    
            # Plot edges
            edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
            az.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
            az.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
            az.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
            az.view_init(20, 30, 0)
            # az.view_init(elev=90, azim=90)
            az.set_box_aspect(None, zoom=1)
            plt.title("Total Energy Density at time stamp: " + str(j) + ", val: " + str(val[p]), fontsize=20)
            plt.xticks(fontsize=18)  # Replace 14 with 12
            plt.yticks(fontsize=18)  # Replace 14 with 12
            z_tick_labels = [r"{0}  ".format(int(label)) for label in az.get_zticks()]  # Add a space to each label
            az.set_zticklabels(z_tick_labels)
            az.w_zaxis.set_tick_params(labelsize=18)
            plt.show()
            # figu.savefig(dir_out + "/tot_contourplot_" + "nx_" + str(nx) + "_time_" + str(j) + "_val_" + str(val[p]) + ".jpg")

        #-------------------------------------------------------------------------------------------------------

    if pot==1:
        for p in range(0,len(val2)):    
    
            figu = plt.figure(figsize=(10, 9))
            az = figu.add_subplot(111, projection='3d')
        
            X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), -np.arange(nz))
        
            kw = {
                'vmin': data.min(),
                'vmax': data.max(),
                'levels': np.linspace(data_array2.min(), data_array2.max(), 10),
            }
        
            for i in range(nz):
                az.contour(
                    X[:, :, i], Y[:, :, i], data_array2[:, :, i],
                    levels=[val2[p]], colors=[plt.cm.jet(p / len(val))], zdir='z', offset=-i,linewidths = 0.2,
                )
        
            for i in range(nx):
                az.contour(
                    X[i, :, :], Y[i, :, :], data_array2[i, :, :],
                    levels=[val2[p]], colors=[plt.cm.jet(p / len(val))], zdir='y', offset=0,linewidths = 0.2,
                )
        
            az.contour(
                data_array2[:, -1, :], Y[:, -1, :], Z[:, -1, :],
                levels=[val2[p]], colors=[plt.cm.jet(p / len(val))], zdir='x', offset=nx,linewidths = 0.2,
            )
        
            # Set limits of the plot from coord limits
            xmin, xmax = X.min(), X.max()
            ymin, ymax = Y.min(), Y.max()
            zmin, zmax = Z.min(), Z.max()
            az.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
        
            # Plot edges
            edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
            az.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
            az.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
            az.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
            az.view_init(20, 30, 0)
            # az.view_init(elev=90, azim=90)
            az.set_box_aspect(None, zoom=1)
            plt.title(f"Potential Energy Density at time: {j*dt:.2f}, val: {val2[p]}", fontsize=20)

            plt.xticks(fontsize=18)  # Replace 14 with 12
            plt.yticks(fontsize=18)  # Replace 14 with 12
            z_tick_labels = [r"{0}  ".format(int(label)) for label in az.get_zticks()]  # Add a space to each label
            az.set_zticklabels(z_tick_labels)
            az.w_zaxis.set_tick_params(labelsize=18)
            # plt.show()
    
            figu.savefig(dir_out + "/pot_contourplot_" + "nx_" + str(nx) + "_time_" + str(j) + "_val_" + str(val2[p]) + ".jpg")
    

    del(data_array)
    del(data_array2)
        
    print(j)    