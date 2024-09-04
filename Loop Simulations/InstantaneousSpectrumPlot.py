import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.fft import fftn, fftshift
import math
from scipy.optimize import curve_fit



def linear_func(x, m, c):
    return m * x + c


dt = 0.05

nt = 2200

start = 300
end = 2400
steps = 100

savespecfreq = 100

dx = dy = dz = 0.25  
nx = ny = nz = dimension = 1024
dumped_nt = 100
dt = 0.05

folders = ["1024_v_2R", "1024_v_5R", "1024_v_0"]#["512_v_2R", "512_v_5R", "512_v_10R", "512_v_0"]
vel = ["$\omega$=1/2R", "$\omega$=1/5R", "$\omega$=0"]#["Velocity = 1/2R", "Velocity = 1/5R", "Velocity = 1/10R", "Velocity = 0"]

n_bin = 200

bin_end = (np.pi)/(dx)
bin_start = (2*np.pi)/(((3)**(1/2))*nx*dx)
epsilon = (bin_end-bin_start)/(n_bin*2)
k_bins = np.linspace(bin_start, bin_end, n_bin)

log_scale = 1
actual_log = 0
compute = 0


time_stepss = np.arange(start, end, steps)

output_directory = rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\out_ro\1024"


shape = (nx, ny, nz)  
ro_a = np.zeros((nt-dumped_nt))

start_index = 5  # Change this to the desired start index
end_index = 100  # Change this to the desired end index


if (compute == 1):
    for folder in folders:
        
        for i in time_stepss:
           
            #ro_a_t = np.loadtxt("data/GifData/ro_Data_Global_expanding_physical_nx512_dampednt482_ntHeld0_seed22_"+str(i)+".txt")
            #ro_a_t_dt = np.loadtxt("data/GifData/ro_Data_Global_expanding_physical_nx512_dampednt482_ntHeld0_seed22_"+str(i-savespecfreq)+".txt")
           
            #gamma = (ro_a_t-ro_a_t_dt)/(savespecfreq*dt)
           
            spec_t = np.loadtxt(rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{folder}\DS\mask1_transform_nx{dimension}_seed22the_iteration"+str(i)+".txt")
            spec_t_dt = np.loadtxt(rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{folder}\DS\mask1_transform_nx{dimension}_seed22the_iteration"+str(i-2*savespecfreq)+".txt")
           
            spec_t = spec_t[:,0]
            spec_t_dt = spec_t_dt[:,0]
        
            tau_t = (1+(i-dumped_nt)*dt)
            tau_t_dt = (1+(i-2*savespecfreq-dumped_nt)*dt)
           
            spec_t = spec_t*(tau_t**4)
            spec_t_dt = spec_t_dt*(tau_t_dt**4)
           
            d_tau = tau_t-tau_t_dt  
           
            F = (spec_t-spec_t_dt)/(4*d_tau)#(1/(gamma))*
            np.save(rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{folder}\DS\non_norm1_F_time_"+str(i)+"_dt_"+str(2*savespecfreq*dt),F)
    
    for folder in folders:
        
        for i in range(dumped_nt+2*savespecfreq, nt, savespecfreq):
           
            #ro_a_t = np.loadtxt("data/GifData/ro_Data_Global_expanding_physical_nx512_dampednt482_ntHeld0_seed22_"+str(i)+".txt")
            #ro_a_t_dt = np.loadtxt("data/GifData/ro_Data_Global_expanding_physical_nx512_dampednt482_ntHeld0_seed22_"+str(i-savespecfreq)+".txt")
           
            #gamma = (ro_a_t-ro_a_t_dt)/(savespecfreq*dt)
           
            spec_t = np.loadtxt(rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{folder}\DS\mask2_transform_nx{dimension}_seed22the_iteration"+str(i)+".txt")
            spec_t_dt = np.loadtxt(rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{folder}\DS\mask2_transform_nx{dimension}_seed22the_iteration"+str(i-2*savespecfreq)+".txt")
           
            spec_t = spec_t[:,0]
            spec_t_dt = spec_t_dt[:,0]
        
            tau_t = (1+(i-dumped_nt)*dt)
            tau_t_dt = (1+(i-2*savespecfreq-dumped_nt)*dt)
           
            spec_t = spec_t*(tau_t**4)
            spec_t_dt = spec_t_dt*(tau_t_dt**4)
           
            d_tau = tau_t-tau_t_dt  
           
            F = (spec_t-spec_t_dt)/(4*d_tau)#(1/(gamma))*
            np.save(rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{folder}\DS\non_norm2_F_time_"+str(i)+"_dt_"+str(2*savespecfreq*dt),F)


for i in time_stepss:
# for i in range(1900, nt, savespecfreq):
    fig, ax = plt.subplots(figsize =(14, 10))
    all_labels = []
    for folder, velocity in zip(folders, vel):
    
    

        storage_mask1 = np.load(rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{folder}\DS\non_norm1_F_time_"+str(i)+"_dt_" + str(2*savespecfreq*dt) + ".npy")
        storage1 = storage_mask1
        Rh = 1
        #plt.show()
        if (log_scale==1):
            plt.plot((k_bins/Rh),(storage1), label = f"Unmasked, {velocity}")
            all_labels.append(f"Unmasked, {velocity}")
        if (actual_log==1):
            plt.plot(np.log(k_bins/Rh),np.log(storage1), label = f"{velocity}")
        
    for folder, velocity in zip(folders, vel):
        
        storage_mask1 = np.load(rf"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 9\neumann_damped\{folder}\DS\non_norm2_F_time_"+str(i)+"_dt_" +str(2*savespecfreq*dt) + ".npy")
        storage1 = storage_mask1
        #plt.show()
        cut = k_bins[start_index:end_index] / Rh
        storage_cut = storage1[start_index:end_index]
        if (log_scale==1):
            plt.plot((k_bins/Rh),(storage1), label = f"Masked, {velocity}")
            all_labels.append(f"Masked, {velocity}")
            
            popt, _ = curve_fit(linear_func, np.log(cut), np.log(storage_cut))
            plt.plot(cut, np.exp(linear_func(np.log(cut), *popt)), label=f'Linear Fit (Gradient: {popt[0]:.2f}),  {velocity}', linestyle='--', linewidth=1.2)
            all_labels.append(f'Linear Fit (Gradient: {popt[0]:.2f}),  {velocity}')
            
        if (actual_log==1):
            plt.plot(np.log(k_bins/Rh),np.log(storage1), label = f"$\phi_0$ Cut, {velocity}")
            
            popt, _ = curve_fit(linear_func, np.log(cut), np.log())
            plt.plot(np.log(cut), linear_func(np.log(cut), *popt), label=f'Linear Fit (Gradient: {popt[0]:.2f})', linestyle='--', linewidth=1.2)
                
            # popt, _ = curve_fit(linear_func, np.log(cut), np.log(storage_cut))
            # plt.plot(cut, np.exp(linear_func(np.log(cut), *popt)), label=f'Linear Fit (Gradient: {popt[0]:.2f}), {velocity}', linestyle='--', linewidth=1.2)
            
    plt.text(0.059, 0.97, f'${nx}^{3}$', transform=ax.transAxes, fontsize=17, verticalalignment='top', horizontalalignment='left')
    plt.text(0.03, 0.92, f'Time={i*0.05}', transform=ax.transAxes, fontsize=17, verticalalignment='top', horizontalalignment='left')
    
    if (log_scale==1):
        plt.yscale("log")
        plt.xscale("log")
    if (actual_log==1):
        plt.xlim(bin_start+1*epsilon,bin_end-1*epsilon)
        plt.ylim(top=12)
    plt.legend(all_labels, fontsize=20, bbox_to_anchor=(1, -0.12), ncol=2)  # Adjust ncol as needed
    plt.xlabel("$|k|$", fontsize = 28)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=17)  # Replace 14 with 12
    plt.yticks(fontsize=17)  # Replace 14 with 12
    ax = plt.gca()
    ax.minorticks_on()
    # plt.legend(fontsize=20)
    ax.tick_params(which='minor', top=True, right=True)
    ax.tick_params(which='major', top=True, right=True)
    ax.tick_params(which='major', length=14, width=1, direction='in')
    ax.tick_params(which='minor', length=6, width=0.7, direction='in')  
    plt.xlim(np.min(k_bins),np.max(k_bins))
    plt.ylim(top=1e5)
    plt.ylabel(r'$\mathcal{F}$', fontsize=32)
    # plt.title(str(i), fontsize = 18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"ro_plot_{nx}_time_{i}.png"), dpi=300)
    plt.show()
    
    print(i)