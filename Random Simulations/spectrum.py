import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.fft import fftn, fftshift
import math

# Specify the directory where the data file is located
dx = 0.25   
dy = 0.25
dz = 0.25
nx = 100
ny = 100
nz = 100

bin_end = (np.pi)/(dx)
bin_start = (2*np.pi)/(((3)**(1/2))*nx*dx)
n_bin =40
shape = (nx, ny, nz)  
#input_directory = 'separate files/'

def fftfreq3D(shape,dx,dy,dz):

    Nx, Ny, Nz = shape

    freqs_x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, dx))
    freqs_y = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, dy))
    freqs_z = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nz, dz))      
    freqs_3D = np.array(np.meshgrid(freqs_x, freqs_y, freqs_z, indexing='ij'))
        
    return freqs_3D

# Load data from the specified input directory
datain = np.loadtxt(r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem7\MPhys\MPhys Folder\test100 W8-9\field_axion_nx100_seed22the_iteration1000.txt")

x_grid = datain[:,0]
y_grid = datain[:,1]
z_grid = datain[:,2]

#Without masking
data_real = datain[:,3]
data_im = np.zeros(len(data_real))

#Masking
# data_real = datain[:,3]*datain[:,4]
# data_im = datain[:,3]*datain[:,5]

y_grid = (y_grid*(1/dy) + (ny)/2)
x_grid = (x_grid*(1/dx) + (nx)/2)
z_grid = (z_grid*(1/dy) + (nz)/2)
# values = data[:, 3]        # Assuming the fourth column contains the function values
# values = values * np.sqrt(2)

data_3d = np.zeros((nx,ny,nz), dtype = complex)

for i in range(0,len(data_im)):
    x = int(x_grid[i])
    y = int(y_grid[i])
    z = int(z_grid[i])
   
    data_3d[x][y][z] = complex(data_real[i] * np.sqrt(2), data_im[i] * np.sqrt(2))

fourier_dat = fftn(data_3d, norm="forward")
fourier_datt = np.fft.fftshift(fourier_dat)
fourier_data = np.abs(fourier_datt)
fourier_data = fourier_data**2

k_space = fftfreq3D(shape, dx,dy,dz)

dkx = abs(k_space[0,89,60,70] - k_space[0,90,60,70]) 

constant_factor = 1/((2*np.pi*nx*dkx)**3)                    #(2 * np.pi)**2 / (8 * np.pi**3 * volume)

k_bins = np.linspace(bin_start, bin_end, num=n_bin)

epsilon = (bin_end-bin_start)/(n_bin*2)
storage = np.zeros((n_bin))

points = np.zeros((n_bin))
sphere = np.zeros((n_bin))

for z in range(0,nz):
    for y in range(0,ny):
        for x in range(0,nx):
            k = k_space[:,x,y,z]
            kx = k[0]
            ky = k[1]
            kz = k[2]
            if((kx != 0) and (kz != 0) and(ky != 0)) :
                mod_k = np.sqrt(kx**2 + ky**2 + kz**2)
            
                #if ((mod_k >= (np.pi / dx)) & (mod_k < ((np.sqrt(3) * np.pi) / dx))):
                for m in range(0, n_bin):
                    if ((mod_k >= ((m)*2*epsilon)) & (mod_k < 2*epsilon*(m+1))):                        
                        # storage[m] = storage[m] + np.abs((mod_k * kz[l] * fourier_data[l] * constant_factor))
                        pre_factor = constant_factor/mod_k
                        solid_angle = (((kx**2 - ky**2)*kz)/(kx**2 + ky**2)) + ky - kx         #      )*)/) )
                        current_point = abs(pre_factor*solid_angle*fourier_data[x][y][z])
                        #print(current_point)
                        storage[m] = storage[m] + current_point #Without solid angle
                            #print(m)
                        # print((mod_k * kz[l] * fourier_data[l] * constant_factor))

fig, ax = plt.subplots(figsize =(7, 4))
plt.plot(k_bins,storage)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("|k|")
plt.xlim(bin_start+1*epsilon,bin_end-1*epsilon)
plt.ylabel("the axion spectrum")
plt.title("axion spectrum after masking with smooth method python")    
plt.show()


min_mod_k_1 = 0.46  # Adjust this based on your requirement
max_mod_k_1 = 3.46 # Adjust this based on your requirement

# Apply the window function to Fourier-transformed data
# for z in range(nz):
#     for y in range(ny):
#         for x in range(nx):
#             k = k_space[:, x, y, z]
#             kx = k[0]
#             ky = k[1]
#             kz = k[2]
#             #if((kx != 0) and (kz != 0) and(ky != 0)) :
#             mod_k_new = np.sqrt(kx**2 + ky**2 + kz**2)
#             if ((mod_k_new >= max_mod_k_1) or (mod_k_new < min_mod_k_1)):
#                 fourier_datt[x][y][z] *= 0
                    
                    
# fourier_datt = np.fft.fftshift(fourier_datt)
# inverse_transform = np.fft.ifftn((fourier_datt), norm="forward")