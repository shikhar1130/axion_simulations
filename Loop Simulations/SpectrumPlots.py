import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.fft import fftn, fftshift
import math
from scipy.optimize import curve_fit


def linear_func(x, m, c):
    return m * x + c

dimension = 1024
dx = 0.25
dt = 0.05
nx = dimension
n_bins = 200

energy_plot_1 = 0
energy_plot_2 = 0
energy_plot_3 = 0
axion_plot = 1
save = 1

start = 1500
end = 1600
interval = 100

bin_end = (np.pi)/(dx)
bin_start = (2*np.pi)/(((3)**(1/2))*nx*dx)
epsilon = (bin_end-bin_start)/(n_bins*2)

time_steps = np.arange(start, end, interval)

folders = [f"{dimension}_network", f"{dimension}_v_2R", f"{dimension}_v_5R", f"{dimension}_v_0"] #[f"{dimension}_network", f"{dimension}_network_300", 
folders2 = [f"{dimension}_v_2R", f"{dimension}_v_5R", f"{dimension}_v_0"]
vel = ["Network", "$\omega$ = 1/2R", "$\omega$ = 1/5R", "$\omega$ = 0"]
vel2 = ["$\omega$ = 1/2R", "$\omega$ = 1/5R", "$\omega$ = 0"]


output_directory = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\out2\1024"
k_bins = np.linspace(bin_start, bin_end, n_bins)

start_ind = 0.3
end_ind = 6

start_index = np.argmin(np.abs(k_bins - start_ind))
end_index = np.argmin(np.abs(k_bins - end_ind))

if energy_plot_1 == 1:
    for i in time_steps:
        plt.figure(figsize=(15, 9))
        
        for fold, vel_label in zip(folders, vel):
            input_directory = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{fold}\DS"
            input_file = os.path.join(input_directory, f"Energy_spectrum_massive_{dimension}_seed22the_iteration{i}.txt")
            data = np.loadtxt(input_file)

            storage = data[:, 0]
            
            plt.plot(k_bins,storage, label=f"{vel_label}")

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
        plt.yscale("log")
        plt.xscale("log")
        plt.text(0.085, 0.95, f"${dimension}^3$", transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.9, f"Time: {i*dt}", transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.legend(fontsize=12, loc="best") 
        plt.xlim(bin_start+1*epsilon,bin_end-1*epsilon)
        # plt.ylim(0, 1e11)
        plt.xlabel("|k|", fontsize=17)
        plt.ylabel("Energy spectrum $(\mathcal{E}_{\chi k})$", fontsize=17)
        if save == 1:
            plt.savefig(os.path.join(output_directory, f"Energy_spectrum_massive_{dimension}_time_step_{i}.png"), dpi=100) 
        # plt.show()
        plt.close()
    
        #-------------------------------------------------------------------------------
if energy_plot_2 == 1:
    for i in time_steps:
        plt.figure(figsize=(15, 9))
        
        for fold, vel_label in zip(folders, vel):
            input_directory = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{fold}\DS"
            input_file = os.path.join(input_directory, f"Energy_spectrum_goldstone_{dimension}_seed22the_iteration{i}.txt")
            data = np.loadtxt(input_file)

            storage = data[:, 0]
            
            plt.plot(k_bins,storage, label=f"{vel_label}")
            
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
        plt.yscale("log")
        plt.xscale("log")
        plt.text(0.085+0.4, 0.95, f"${dimension}^3$", transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05+0.4, 0.9, f"Time: {i*dt}", transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.xlim(bin_start+1*epsilon,bin_end-1*epsilon)
        # plt.ylim(0, 1e11)
        plt.xlabel("|k|", fontsize=17)
        plt.ylabel(r"Energy spectrum $(\mathcal{E}_{\theta k})$", fontsize=17)
        plt.legend(fontsize=12, loc="best")   
        if save == 1:
            plt.savefig(os.path.join(output_directory, f"Energy_spectrum_goldstone_{dimension}_time_step_{i}.png"), dpi=100) 
        # plt.show()
        plt.close()
    
if energy_plot_3 == 1:
    for i in time_steps:
        plt.figure(figsize=(15, 15))
        
        for fold, vel_label in zip(folders, vel):
            input_directory = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{fold}\DS"
            input_file = os.path.join(input_directory, f"Energy_spectrum_{dimension}_seed22the_iteration{i}.txt")
            data = np.loadtxt(input_file)

            storage = data[:, 0]
            
            plt.plot(k_bins,storage, label=f"{vel_label}")
            
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
        plt.yscale("log")
        plt.xscale("log")
        plt.text(0.085+0.4, 0.95, f"${dimension}^3$", transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05+0.4, 0.9, f"Time: {i*dt}", transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.xlim(bin_start+1*epsilon,bin_end-1*epsilon)
        # plt.ylim(0, 1e11)
        plt.xlabel("|k|", fontsize=17)
        plt.ylabel(r"Kinetic Energy spectrum $(\mathcal{K})$", fontsize=17)
        plt.legend(fontsize=12, loc="best")   
        if save == 1:
            plt.savefig(os.path.join(output_directory, f"Energy_spectrum_{dimension}_time_step_{i}.png"), dpi=100) 
        plt.show()
        plt.close()

        #---------------------------------------------------------------------------------
if axion_plot == 1:
    for i in time_steps:
        plt.figure(figsize=(14, 10))
    
        all_labels = []  # Store all labels for legend
    
        for fold, vel_label in zip(folders, vel):
            input_directory = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{fold}\DS"
            input_file = os.path.join(input_directory, f"mask1_transform_nx{dimension}_seed22the_iteration{i}.txt")
            try:
              data = np.loadtxt(input_file)
              storage = data[:, 0]
              plt.plot(k_bins, storage, label=f"Unmasked, {vel_label}")
              all_labels.append(f"Unmasked, {vel_label}")
            except FileNotFoundError:
                print(f"File not found for {fold}, {vel_label}. Continuing to the next curve...")
    
    
        for fold, vel_label in zip(folders2, vel2):
            input_directory = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{fold}\DS"
            input_file = os.path.join(input_directory, f"mask2_transform_nx{dimension}_seed22the_iteration{i}.txt")
            try:
                data = np.loadtxt(input_file)
                storage = data[:, 0]
                plt.plot(k_bins, storage, label=f"Masked, {vel_label}")
                all_labels.append(f"Masked, {vel_label}")
                
                cut = k_bins[start_index:end_index]
                storage_cut = storage[start_index:end_index]
                
                # Perform the curve fitting
                popt, _ = curve_fit(linear_func, np.log(cut), np.log(storage_cut))
                plt.plot(cut, np.exp(linear_func(np.log(cut), *popt)), label=f'Linear Fit (Gradient: {popt[0]:.2f}), {vel_label}', linestyle='--', linewidth=1.2)
                all_labels.append(f'Linear Fit (Gradient: {popt[0]:.2f}), {vel_label}')
               
            except FileNotFoundError:
                print(f"File not found for {fold}, {vel_label}. Continuing to the next curve...")
           
            
    
        # Plot finished, configure legend
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=17)  # Replace 14 with 12
        plt.yticks(fontsize=17)  # Replace 14 with 12
        ax = plt.gca()
        ax.minorticks_on()
    
        # Set legend location and adjust ncol for horizontal distribution
        plt.legend(all_labels, fontsize=20, bbox_to_anchor=(1, -0.1), ncol=2)  # Adjust ncol as needed
    
        ax.tick_params(which='minor', top=True, right=True)
        ax.tick_params(which='major', top=True, right=True)
        ax.tick_params(which='major', length=14, width=1, direction='in')
        ax.tick_params(which='minor', length=6, width=0.7, direction='in')
        
        plt.yscale("log")
        plt.xscale("log")
        plt.text(0.085, 0.95, f"${dimension}^3$", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
        plt.text(0.05, 0.9, f"Time: {i*dt}", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
        plt.xlim(bin_start+1*epsilon,bin_end-1*epsilon)
        plt.ylim(0, 5e4)
        plt.xlabel("|k|", fontsize=22)
        plt.ylabel(r"Axion spectrum $ \left (\frac{\partial \rho_{a}}{\partial k} \right ) $", fontsize=22)
        plt.tight_layout()
        if save == 1:
          plt.savefig(os.path.join(output_directory, f"Axion_spectrum_{dimension}_time_step_{i}.png"), dpi=300)
        plt.show()
        plt.close()
