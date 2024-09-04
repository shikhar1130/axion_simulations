import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy
from scipy.fft import fftn, fftshift
import math
from matplotlib import colors
# Specify the directory where the data file is located
dx = 0.25   
dy = 0.25
dz = 0.25

nx = 256
ny = 256
nz = 256

lengthCutOff = ((dx**2)*3)**(0.5)
stringTotalLength = np.zeros((2000))
averageCurv = np.zeros((2000))
curv_rich = np.zeros((2000))
varCurv = np.zeros((2000))
save_curve = "average_curv.txt"
save_length = "total_length.txt"
save_varcurv = "var_curv.txt"
save_rich = "curv_rich.txt"

input_directory = r"C:\Users\shikh\OneDrive - The University of Manchester\Docs\UOM\UoM\Course\Sem8\MPhys\MPhys Project\Week 1\expanding_universe_testing\conformal_time_fixed\256\GifData"

for i in range(400, 2000, 25):
    # Create the full path for the input file
    input_file = os.path.join(input_directory, f"gifStringPosData_Global_expanding_physical_nx256_dampednt400_ntHeld0_seed22_{i}.txt")

    # Load the data from the specified file
    stringData = np.loadtxt(input_file)
# Convert from indices to physical positions
    stringData2 = stringData
    stringData2[:,0] += 0.5*(nx-1)*dx
    stringData2[:,1] += 0.5*(ny-1)*dy
    stringData2[:,2] += 0.5*(nz-1)*dz 
    for k in range(0,len(stringData2)):
        for l in range(0,3):
            if (stringData2[k][l] >128):
                stringData2[k][l] = stringData2[k][l]-128
               
    
    eps = 10**-6 # To stop tree construction issues related to input data > periodic boundaries
    tree = scipy.spatial.cKDTree(stringData2[:,0:3],boxsize=[nx*dx+eps,ny*dy+eps,nz*dz+eps])
           
    #neighbours = tree.query_ball_point(stringData[:,0:3],np.sqrt(dx**2+dy**2+dz**2))
    neighbours = tree.query(stringData2[:,0:3],k=[2,3])
           
    # Curvature calculations:
    # Already have the lengths of two sides of the triangle formed between query point and 2 neighbours:
    a = neighbours[0][:,0]
    b = neighbours[0][:,1]
           
    # Calculate the total length of string in the simulation.
    # Sum all neighbour distances for each point and then divide by two at the end to account for double counting.
    cutoffLogic = (a<lengthCutOff) & (b<lengthCutOff)
    
    a_valid = a[cutoffLogic]
    b_valid = b[cutoffLogic]
    string_net = 0.5 * (np.sum(a_valid) + np.sum(b_valid))
    
    # string_net = 0.5*(np.sum(a[a<lengthCutOff]) +np.sum(b[b<lengthCutOff]))
    stringTotalLength[i] = string_net
           
    # Need to calculate the distance between the two neighbours though
    c = np.zeros(len(stringData2[:,0]))
           
    # Logic to determine whether to account for periodicity or not
    xLogic = abs(stringData2[neighbours[1][:,0],0]-stringData2[neighbours[1][:,1],0]) >= 0.5*nx*dx
    yLogic = abs(stringData2[neighbours[1][:,0],1]-stringData2[neighbours[1][:,1],1]) >= 0.5*ny*dy
    zLogic = abs(stringData2[neighbours[1][:,0],2]-stringData2[neighbours[1][:,1],2]) >= 0.5*nz*dz
           
    c[~xLogic] += (stringData2[neighbours[1][~xLogic,0],0] - stringData2[neighbours[1][~xLogic,1],0])**2
    c[xLogic] += (nx*dx - abs(stringData2[neighbours[1][xLogic,0],0] - stringData2[neighbours[1][xLogic,1],0]) )**2
           
    c[~yLogic] += (stringData2[neighbours[1][~yLogic,0],1] - stringData2[neighbours[1][~yLogic,1],1])**2
    c[yLogic] += (ny*dy - abs(stringData2[neighbours[1][yLogic,0],1] - stringData2[neighbours[1][yLogic,1],1]) )**2
           
    c[~zLogic] += (stringData2[neighbours[1][~zLogic,0],2] - stringData2[neighbours[1][~zLogic,1],2])**2
    c[zLogic] += (nz*dz - abs(stringData2[neighbours[1][zLogic,0],2] - stringData2[neighbours[1][zLogic,1],2]) )**2
           
    c = np.sqrt(c) # Convert to the actual distance
          
    # Calculate the curvature by fitting to a circle and taking inverse of the radius.
    curv = np.sqrt( (a+b+c)*(-a+b+c)*(a-b+c)*(a+b-c) )/(a*b*c);
    np.save("curvature_"+str(i)+"txt",curv)
    
    common_cutoff_logic = (a < lengthCutOff) & (b < lengthCutOff)

    # Extract valid elements from a and b based on common cutoff logic
    a_valid = a[common_cutoff_logic]
    b_valid = b[common_cutoff_logic]
    
    length = 0.5*(a_valid + b_valid)
    np.save("length_"+str(i)+"txt",length)
    curv[np.isnan(curv)] = 0
    curv[np.isinf(curv)] = 0
    curv_rich[i] = ((np.sum(0.5*(a+b)*curv))/string_net)
    averageCurv[i] = np.average(curv)
    varCurv[i] = np.var(curv)
    print(i)
np.savetxt(save_curve,averageCurv)
np.savetxt(save_length,stringTotalLength)
np.savetxt(save_varcurv,varCurv)
np.save(save_rich, curv_rich)