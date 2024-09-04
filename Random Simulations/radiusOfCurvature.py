import numpy as np
import scipy.spatial   

nx = 101
ny = 101
nz = 101
dx = 0.5
dy = 0.5
dz = 0.5
lengthCutOff = 1500   #Will ask Steven about this value today
i = 1               #Will ask Steven about this value today

# Read data from the file
with open('gifStringPosData_global_PRS_dx0p25_nx256_seed42_0.txt', 'r') as file:
    lines = file.readlines()
    
# Create an empty list to store the string data
stringData = []

# Parse the lines and store values in stringData
for line in lines:
    values = line.strip().split()
    if len(values) >= 3:
        x, y, z = float(values[0]), float(values[1]), float(values[2])
        stringData.append([x, y, z])

# Convert stringData to a NumPy array
stringData = np.array(stringData)

stringTotalLength = np.zeros(3) 
averageCurv = np.zeros(3)        
varCurv = np.zeros(3)  

# Convert from indices to physical positions
stringData2 = np.array(stringData[:,0:3])
stringData2[:,0] += 0.5*(nx-1)*dx
stringData2[:,1] += 0.5*(ny-1)*dy
stringData2[:,2] += 0.5*(nz-1)*dz 

inside_box = (
    (stringData2[:, 0] >= 0) &
    (stringData2[:, 0] < nx * dx) &
    (stringData2[:, 1] >= 0) &
    (stringData2[:, 1] < ny * dy) &
    (stringData2[:, 2] >= 0) &
    (stringData2[:, 2] < nz * dz)
)

stringData2 = stringData2[inside_box]

# Reconstruct the cKDTree with the adjusted stringData2
eps = 10**-6 # To stop tree construction issues related to input data > periodic boundaries
tree = scipy.spatial.cKDTree(stringData2, boxsize=[nx * dx + eps, ny * dy + eps, nz * dz + eps])

#tree = scipy.spatial.cKDTree(stringData2[:,0:3],boxsize=[nx*dx+eps,ny*dy+eps,nz*dz+eps])
       
#neighbours = tree.query_ball_point(stringData[:,0:3],np.sqrt(dx**2+dy**2+dz**2))
neighbours = tree.query(stringData2[:,0:3],k=[2,3])
       
# Curvature calculations:
# Already have the lengths of two sides of the triangle formed between query point and 2 neighbours:
a = neighbours[0][:,0]
b = neighbours[0][:,1]

# Calculate the total length of string in the simulation.
# Sum all neighbour distances for each point and then divide by two at the end to account for double counting.
cutoffLogic = (a<lengthCutOff) & (b<lengthCutOff)
stringTotalLength[i] = 0.5*(np.sum(a[a<lengthCutOff]) +np.sum(b[b<lengthCutOff]))
       
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
curv[np.isnan(curv)] = 0
curv[np.isinf(curv)] = 0
averageCurv[i] = np.average(curv)
varCurv[i] = np.var(curv)

# Calculate the radius of curvature
radius_of_curvature = 1 / curv

# Print the radius of curvature
print("Radius of curvature:", radius_of_curvature)


# Store the values in an output file
output_file = 'radius_of_curvature_out.txt'
with open(output_file, 'w') as out_file:
    for value in radius_of_curvature:
        out_file.write(f"{value}\n")