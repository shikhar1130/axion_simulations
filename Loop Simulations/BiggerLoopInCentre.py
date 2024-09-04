import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy
import os

dimension = 512
spacing = 0.25

start = 500
end = 505
interval = 10

save = 0
show = 1

dx = dy = dz = spacing
nx = ny = nz = dimension

eps = 10**-6

dir_in = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 4\loop_in_middle\{dimension}\GifData"
dir_out = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 4\loop_in_middle\{dimension}\out"


lengthCutOff = (dx**2+dy**2+dz**2)**(0.5)
param = 0 # involved in curvature
for j in range(start, end, interval):
    try:
        # calculate curvature and save it
        input_file = os.path.join(dir_in, f"gifStringPosData_Global_static_physical_nx{dimension}_dampednt0_ntHeld0_seed22_{j}.txt")
        stringData = np.loadtxt(input_file)
    
# val = np.linspace(0,2,21)


        if len(stringData) == 0:
          print(f"Skipping {input_file} as it is empty.")
          continue # Skip to the next iteration
    
        # Identify outer and inner points based on a distance threshold
        # threshold = .26 * nx * dx  # Adjust this threshold as needed
        # outer_points = ((stringData[:, 0] < -threshold) | (stringData[:, 0] > threshold)) | ((stringData[:, 2] < -threshold) | (stringData[:, 2] > threshold))
        # inner_points = ~outer_points
        # stringData = stringData[outer_points]


        stringData2 = np.copy(stringData[:,0:3])
        stringData2[:, 0] += 0.5 * (nx - 1) * dx
        stringData2[:, 1] += 0.5 * (ny - 1) * dy
        stringData2[:, 2] += 0.5 * (nz - 1) * dz


        mask = stringData2[:, 0] > (nx-1)*dx/2
        stringData2[mask, 0] -= (nx-1)*dx
    
        mask2 = stringData2[:, 2] > (nz-1)*dz/2
        stringData2[mask2, 2] -= (nz-1)*dz
        

        stringData2[:,0] += (nx)*dx/2
        stringData2[:,2] += (nz)*dz/2
        

        # stringData2 = stringData2 % box_size
        
        # # Shift inner points to the corners while keeping their relative positions
        # corner_shift = (nx - 1) * dx / 2
        # shifted_inner_points = stringData[inner_points] + corner_shift
        # stringData[inner_points] = shifted_inner_points


                    
        tree = scipy.spatial.cKDTree(stringData2[:,0:3],boxsize=[2*nx*dx+eps,2*ny*dy+eps,2*nz*dz+eps])    
        neighbours = tree.query(stringData2[:,0:3],k=[2,3])    
        a = neighbours[0][:,0]
        b = neighbours[0][:,1]    
        cutoffLogic = (a<lengthCutOff) & (b<lengthCutOff)
        #stringTotalLength[i] = 0.5*(np.sum(a[a<lengthCutOff]) +np.sum(b[b<lengthCutOff]))           
        c = np.zeros(len(stringData2[:,0])) 
        
        xLogic = abs(stringData2[neighbours[1][:,0],0]-stringData2[neighbours[1][:,1],0]) >= 0.5*nx*dx
        yLogic = abs(stringData2[neighbours[1][:,0],1]-stringData2[neighbours[1][:,1],1]) >= 0.5*ny*dy
        zLogic = abs(stringData2[neighbours[1][:,0],2]-stringData2[neighbours[1][:,1],2]) >= 0.5*nz*dz   
        
        c[~xLogic] += (stringData2[neighbours[1][~xLogic,0],0] - stringData2[neighbours[1][~xLogic,1],0])**2
        c[xLogic] += (nx*dx - abs(stringData2[neighbours[1][xLogic,0],0] - stringData2[neighbours[1][xLogic,1],0]) )**2
               
        c[~yLogic] += (stringData2[neighbours[1][~yLogic,0],1] - stringData2[neighbours[1][~yLogic,1],1])**2
        c[yLogic] += (ny*dy - abs(stringData2[neighbours[1][yLogic,0],1] - stringData2[neighbours[1][yLogic,1],1]) )**2
               
        c[~zLogic] += (stringData2[neighbours[1][~zLogic,0],2] - stringData2[neighbours[1][~zLogic,1],2])**2
        c[zLogic] += (nz*dz - abs(stringData2[neighbours[1][zLogic,0],2] - stringData2[neighbours[1][zLogic,1],2]) )**2           
        c = np.sqrt(c)    
        curv = np.sqrt( (a+b+c)*(-a+b+c)*(a-b+c)*(a+b-c) )/(a*b*c);
        curv[np.isnan(curv)] = 0
        curv[np.isinf(curv)] = 0 
    
    
        plt.figure(figsize=(11, 9))
        ax = plt.axes(projection='3d')
        # ax.set_xlim3d(-(nx-1)*dx/2,(nx+1)*dx/2) #Assymetric due to the periodic boundary positions (string can be between boundaries)
        # ax.set_ylim3d(-(ny-1)*dy/2,(ny+1)*dy/2)
        # ax.set_zlim3d(-(nz-1)*dz/2,(nz+1)*dz/2)
        
        ax.set_xlim3d(0, (nx-1)*dx)
        ax.set_ylim3d(0, (ny-1)*dy)
        ax.set_zlim3d(0, (nz-1)*dz)

        ax.set_xlabel('$x$', fontsize=20)
        ax.set_ylabel('$y$', fontsize=20)
        ax.set_zlabel('$z$', fontsize=20)
        
        z_ticks = ax.get_zticks()
        ax.set_zticks(z_ticks)
        z_tick_labels = [r"{0}  ".format(int(label)) for label in z_ticks]  # Add a space to each label
        ax.set_zticklabels(z_tick_labels)
                
        z_tick_labels = [r"{0}  ".format(int(label)) for label in ax.get_zticks()]  # Add a space to each label
        ax.set_zticklabels(z_tick_labels)
        ax.zaxis.set_tick_params(labelsize=14)
        
        # Set colour map properties
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=0,vmax=0.2)
            
        
        ax.scatter3D(stringData2[cutoffLogic,0],stringData2[cutoffLogic,1],stringData2[cutoffLogic,2],s=3,c=cmap(norm(curv[cutoffLogic])))
        ax.legend(["Time stamp: " + str(j)], fontsize=18)
        # ax.view_init(elev=0, azim=90)
        
        if (show == 1):
            plt.show()
        
        
        #plt.title("Time step: "+str(j*inte + ofset)+", energy density on the surface: "+str(val[p]))
        
        plot_3 = os.path.join(dir_out, "plot_of_surf_and_curv_timestep_" + str(j) + ".png")
        
        if (save == 1):
            plt.savefig(plot_3, dpi=300)
        
        

        del(stringData)
            
        print(j)    
    
    except OSError:
        print(f"Skipping {input_file} as it is empty.")
