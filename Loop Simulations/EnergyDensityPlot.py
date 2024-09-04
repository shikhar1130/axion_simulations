# Plot of energy density:



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy

dx = 0.25
dy = 0.25
dz = 0.25
ofset = 0
dv = dx*dy*dz

nx = 256
ny = 256
nz = 256
eps = 10**-6
inte = 10
dir_out = r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 5\energy_density\256\out"
dir_in = r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 5\energy_density\256\DS"
dir_in2 = r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 4\loop_in_middle\256\GifData"
pos_max = nx/2

lengthCutOff = (dx**2+dy**2+dz**2)**(0.5)
param = nx*dx*.5
# val = np.linspace(0,2,21)
val = [0.4]


for j in range(290, 300, inte):
 #calculate energy and save it   
 
    input_file = dir_in + "/Energydensity_Global_static_physical_nx256_dampednt0_ntHeld0_seed22_"+str(j)+".txt"
    data = np.loadtxt(input_file)
    print("Data loaded.")
    energy_pot = 0
    energy_tot = 0
    
    length = len(data)
    
    for i in range(0,length):
        
        energy_pot = energy_pot + data[i][3]
        energy_tot = energy_tot + data[i][4]
    
    energy = np.array([[energy_pot], [energy_tot]])    
        
    # out1 = dir_out +  "int_energy"+str(j)+".txt"        
    
    #np.savetxt(out1, energy)
    
 #transform coordinates and save a np array of energy densities
 
    data_array = np.zeros((nx,ny,nz)) 
    for i in range(0,length):
        x = int((data[i][0] + param)/dx)
        y = int((data[i][1] + param)/dy)
        z = int((data[i][2] + param)/dz) 
        value_tot = (data[i][4]) #total energy density
        value_pot = (data[i][3]) #potential energy density
        data_array[x][y][z] = value_tot
    
    print("Processing done (energy).")
