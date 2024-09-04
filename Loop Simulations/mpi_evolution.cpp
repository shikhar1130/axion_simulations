#define _USE_MATH_DEFINES
#include <complex>
#include <cmath>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <vector>
#include <mpi.h>
#include <random>
#include <algorithm>
using namespace std;

//to be addjusted
//grid param
const long long int nx = 1024;
const long long int ny = 1024;
const long long int nz = 1024;
const int nt = 3500; // damped_nt + nt required to reach light crossing time. nt required for sim to end at light crossing time is nx*dx/(2*dt).
const double dx = 0.7;
const double dy = 0.7;
const double dz = 0.7;
const double dt = 0.14;

//Boundary conditions (Anything else apart from "fixed" or "absorbing" or "neumann" will use periodic boundary conditions currently)
std::string xBC = "neumann";
std::string yBC = "neumann";
std::string zBC = "neumann";

//model param
const double lambda = 1;
const double eta = 1;
const double m_chi = sqrt(2 * lambda) * eta; // for vachaspati energy spectrum
const double g = 0*sqrt(0.5*lambda); // fracBPS*g_BPS
const int damped_nt = 100; // Number of time steps for which damping is imposed. Useful for random initial conditions
const double dampFac = 0.5; // magnitude of damping term, unclear how strong to make this
const int ntHeld = 0; // Hold fields fixed (but effectively continue expansion) for this number of timesteps. Attempting to get the network into the scaling regime.
const double alpha = 2; // Factor multiplying hubble damping term for use in PRS algorithm. alpha = #dims has been claimed to give similar dynamics without changing string width.
                        // Standard dynamics is alpha = #dims -1
const double beta_phi = 2; // scale factor^beta_phi is the factor that multiplies the potential contribution to the phi EoMs. Standard is 2, PRS is 0.
const double beta_g = 2; // scale factor^beta_g is the factor that multiplies the contribution of the current to the gauge EoMs. Standard is 2, PRS is 0.
const double scaling = 0; // Power law scaling of the scale factor. Using conformal time so rad dom is gamma=1 while matter dom is gamma=2. gamma=0 returns a static universe

///////////////////////////////////////////// Output directory ////////////////////////////////////////////////////
string dir_path = "/home/agarwal/1024_v_2R_0.7";
//------------------------------------------------------------------------------------------------------------------


///////////////////////////////////////////// To output any frames ////////////////////////////////////////////////
const bool frame_out = false; //do you want any frames to be outputted?
const int output_mode = 0;  // 1 - full output(x,y,z,axion,real,im) 0 - (axion,real,im)
std::vector<int> frames = {800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000};
//------------------------------------------------------------------------------------------------------------------


///////////////////////////////////////////// Energy //////////////////////////////////////////////////////////////
const bool calcEnergy = true;
//------------------------------------------------------------------------------------------------------------------


//////////////////////////////////////// Energy spectrum Villadoro/////////////////////////////////////////////////
const bool calc_energy_spectr_direct = false;
const int saveenergyspecfreq = 200; //saving frequency for energy spectrum
const int n_bin_energy_spectr = 200;
//------------------------------------------------------------------------------------------------------------------

///////////////////////////////////////// Energy spectrum Vachaspati ///////////////////////////////////////////////
const bool calc_energy_spectr_indirect = false;
const int saveenergyspecfreq1 = 200; //saving frequency for energy spectrum
//------------------------------------------------------------------------------------------------------------------


///////////////////////////////////////////// Energy density //////////////////////////////////////////////////////
const bool calc_energy_density = false; // Do not turn on for larger grid sizes (Large output files)
const bool output_energy_all = false;
const bool output_energy_total_potential = true;
const int energyDensityFreq = 10;
//------------------------------------------------------------------------------------------------------------------


///////////////////////////////////////////// String detection ////////////////////////////////////////////////////
const bool stringDetect = true; // true if you want the code to find which faces the strings pass through. May try to calculate string length later too.
const int saveFreq = 5; //for string positions
//------------------------------------------------------------------------------------------------------------------


///////////////////////////////////////////// Axion spectrum /////////////////////////////////////////////////////
const bool calc_spectr = false;
const bool ro_a = false;
const int savespecfreq = 100; //saving frequency for spectrum
bool method_1 = true; // smooth masking
bool method_2 = true; // window masking
bool method_3 = false; //unmasked spectrum
bool method_4 = false; //doubly masked spectrum 
double treshold = 0.996; // square of magnitude treshold
double treshold_d = 0.996;
const int n_bin = 200;
//------------------------------------------------------------------------------------------------------------------


const string outTag = "Global_static_physical";
const int seed = 22;

// Never adjusted but useful to define
double bin_start  = (2 * M_PI) / double(sqrt(3) * nx * dx);
double bin_end = (M_PI) / (dx);
double epsilon_energy = (bin_end - bin_start) / (n_bin_energy_spectr * 2);
double epsilon = (bin_end - bin_start) / (n_bin * 2);
double d_k = (2 * M_PI) / (nx * dx);
double constant_factor = d_k * d_k / pow((2 * M_PI * nx * dx), 3);
double constant_factor_energy = 1 / pow((nx * dx), 3);
const int countRate = 10;
const int nts = 2; // Number of time steps saved in data arrays
const double pi = 4.0*atan(1.0);
const long long int nPos = nx*ny*nz;
const bool expandDamp = false; // If true then the universe expands during the damping regime.
const string ic_type = "data"; // Current options are data, stationary data and random
const bool detectBuffer = true; // true if you want to don't want the code to find strings during damping process. Usually because random ic means you will find A LOT of strings --> memory issues.
bool stringsExist = true;
int main(int argc, char ** argv){
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    long long int chunk = nPos/size;
    long long int chunkRem = nPos - size*chunk;

    long long int coreSize;
	if(rank>=chunkRem){ coreSize = chunk; }
	else{ coreSize = chunk+1; }

	long long int coreStart, coreEnd;
    if(rank < chunkRem){ coreStart = rank*(chunk+1); coreEnd = (rank+1)*(chunk+1); }
    else{ coreStart = rank*chunk+chunkRem; coreEnd = (rank+1)*chunk+chunkRem; }
    long long int frontHaloSize, backHaloSize, nbrFrontHaloSize, nbrBackHaloSize, remFront, remBack;
    remFront = coreStart%(ny*nz);
    remBack = coreEnd%(ny*nz);
    if(remFront==0){ 
    	frontHaloSize = ny*nz;
    	nbrBackHaloSize = ny*nz;

    } else{

    	frontHaloSize = ny*nz + remFront;
    	nbrBackHaloSize = 2*ny*nz - remFront;

    }

    if(remBack==0){

    	backHaloSize = ny*nz;
    	nbrFrontHaloSize = ny*nz;

    } else{

    	backHaloSize = 2*ny*nz - remBack;
    	nbrFrontHaloSize = ny*nz + remBack;

    }

    long long int totSize = frontHaloSize + coreSize + backHaloSize;

    long long int dataStart = coreStart-frontHaloSize;
    long long int dataEnd = coreEnd+backHaloSize;
    // Warnings

    if(rank==0){

    	if(size==1){ cout << "Warning: Only one processor being used. This code is not designed for only one processor and may not work." << endl; }
    	if(chunk<ny*nz){ cout << "Warning: Chunk size is less than the minimum halo size (i.e chunk neighbour data). Code currently assumes this is not the case so it probably won't work." << endl; }

    }

	vector<double> phi(2*2*totSize, 0.0), theta(3*2*totSize, 0.0);//, phitt(2*coreSize, 0.0), thetatt(3*coreSize, 0.0);//, energydensity(coreSize, 0.0), gaussDeviation(coreSize, 0.0);
	double phixx,phiyy,phizz,phitx,phity,phitz,localEnergy,deviationParameter,damp,phit[2],phiMagSqr,phiMagSqr_past,curx,cury,curz,Fxy_y,Fxz_z,Fyx_x,Fyz_z,Fzx_x,Fzy_y,localSeed,phitt[2],thetatt[3],thetat[3],phix[2],phiy[2],phiz[2],
		   Fxy,Fxz,Fyz,FCont,thetaDotCont,x0,y0,z0;
	long long int i,j,k,p,f,TimeStep,gifStringPosFrame,tNow,tPast,counter,comp,imx,ipx,imy,ipy,imz,ipz,ipxmy,ipxmz,imxpy,ipymz,imxpz,imypz;
	int c[2] = {1,-1}; // Useful definition to allow covariant deviative to be calculated when looping over components.

	struct timeval start, end;
	if(rank==0){ gettimeofday(&start, NULL); }

    //string file_path = __FILE__;
    //file_path.substr(0,file_path.find_last_of('/'));
    stringstream ss;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); // Allows all other processes to start once user input has been received.

    string icPath = dir_path + "/Data/ic.txt";
    string finalFieldPath = dir_path + "/Data/finalField.txt";
    string valsPerLoopPath = dir_path + "/Data/valsPerLoop_" + outTag + "_nx" + to_string(nx) + "_dampednt" + to_string(damped_nt) + "_ntHeld" + to_string(ntHeld) + "_seed" + to_string(seed) + ".txt";

    ifstream ic (icPath.c_str());
    ofstream finalField (finalFieldPath.c_str());
    ofstream valsPerLoop (valsPerLoopPath.c_str());
   
    x0 = 0.5*(nx-1);
    y0 = 0.5*(ny-1);
    z0 = 0.5*(nz-1);

    double wasteData[5];

    if(ic_type=="stationary data"){

    	for(i=0;i<nPos;i++){

    		// Only assign it to the local array if this point belongs to the core or halo. Otherwise just waste it. Use modulus operator to deal with periodicity
    		// Uses index calculation totSize*(nTimeSteps*component + TimeStep) + pos

    		if(dataStart<0){ // Need to deal with periodicity at the start of the data array

    			if(i>=(dataStart+nPos)%nPos){ // Front halo

    				long long int arrayPos = i-(dataStart+nPos)%nPos;

    				ic >> phi[arrayPos] >> phi[totSize*nts+arrayPos] >> theta[arrayPos] >> theta[totSize*nts+arrayPos] >> theta[totSize*nts*2+arrayPos];

    				// Second time step is equal to the first

    				phi[totSize+arrayPos] = phi[arrayPos];
    				phi[totSize*(nts+1)+arrayPos] = phi[totSize*nts+arrayPos];
    				theta[totSize+arrayPos] = theta[arrayPos];
    				theta[totSize*(nts+1)+arrayPos] = theta[totSize*nts+arrayPos];
    				theta[totSize*(nts*2+1)+arrayPos] = theta[totSize*nts*2+arrayPos];

    			} else if(i<dataEnd){ // The rest of the data

    				long long int arrayPos = i+frontHaloSize; // Shift across to account for the front halo at the start

    				ic >> phi[arrayPos] >> phi[totSize*nts+arrayPos] >> theta[arrayPos] >> theta[totSize*nts+arrayPos] >> theta[totSize*nts*2+arrayPos];

    				phi[totSize+arrayPos] = phi[arrayPos];
    				phi[totSize*(nts+1)+arrayPos] = phi[totSize*nts+arrayPos];
    				theta[totSize+arrayPos] = theta[arrayPos];
    				theta[totSize*(nts+1)+arrayPos] = theta[totSize*nts+arrayPos];
    				theta[totSize*(nts*2+1)+arrayPos] = theta[totSize*nts*2+arrayPos];

    			} else{ ic >> wasteData[0] >> wasteData[1] >> wasteData[2] >> wasteData[3] >> wasteData[4]; } // Don't need these so waste them into an unused variable 


    		} else if(dataEnd>nPos){ // Need to deal with periodicity at the end of the data array

    			if(i>=dataStart){ // All of the array except for the back halo

    				long long int arrayPos = i-dataStart;

					ic >> phi[arrayPos] >> phi[totSize*nts+arrayPos] >> theta[arrayPos] >> theta[totSize*nts+arrayPos] >> theta[totSize*nts*2+arrayPos];

    				phi[totSize+arrayPos] = phi[arrayPos];
    				phi[totSize*(nts+1)+arrayPos] = phi[totSize*nts+arrayPos];
    				theta[totSize+arrayPos] = theta[arrayPos];
    				theta[totSize*(nts+1)+arrayPos] = theta[totSize*nts+arrayPos];
    				theta[totSize*(nts*2+1)+arrayPos] = theta[totSize*nts*2+arrayPos]; 				

    			} else if(i<dataEnd%nPos){ // The back halo

    				long long int arrayPos = i+coreSize+frontHaloSize;

    				ic >> phi[arrayPos] >> phi[totSize*nts+arrayPos] >> theta[arrayPos] >> theta[totSize*nts+arrayPos] >> theta[totSize*nts*2+arrayPos];

    				phi[totSize+arrayPos] = phi[arrayPos];
    				phi[totSize*(nts+1)+arrayPos] = phi[totSize*nts+arrayPos];
    				theta[totSize+arrayPos] = theta[arrayPos];
    				theta[totSize*(nts+1)+arrayPos] = theta[totSize*nts+arrayPos];
    				theta[totSize*(nts*2+1)+arrayPos] = theta[totSize*nts*2+arrayPos]; 

    			} else{ ic >> wasteData[0] >> wasteData[1] >> wasteData[2] >> wasteData[3] >> wasteData[4]; }

    		} else{ // In the middle of the array so don't need to deal with periodicity

    			if(i>=dataStart and i<dataEnd){

    				long long int arrayPos = i-dataStart;

    				ic >> phi[arrayPos] >> phi[totSize*nts+arrayPos] >> theta[arrayPos] >> theta[totSize*nts+arrayPos] >> theta[totSize*nts*2+arrayPos];

    				phi[totSize+arrayPos] = phi[arrayPos];
    				phi[totSize*(nts+1)+arrayPos] = phi[totSize*nts+arrayPos];
    				theta[totSize+arrayPos] = theta[arrayPos];
    				theta[totSize*(nts+1)+arrayPos] = theta[totSize*nts+arrayPos];
    				theta[totSize*(nts*2+1)+arrayPos] = theta[totSize*nts*2+arrayPos]; 

    			} else{ ic >> wasteData[0] >> wasteData[1] >> wasteData[2] >> wasteData[3] >> wasteData[4]; }

    		}

    	}

    } else if(ic_type=="data"){

    	for(TimeStep=0;TimeStep<2;TimeStep++){
    		for(i=0;i<nPos;i++){

    			if(dataStart<0){ // Need to deal with periodicity at the start of the data array

	    			if(i>=(dataStart+nPos)%nPos){ // Front halo

	    				long long int arrayPos = i-(dataStart+nPos)%nPos;

	    				ic >> phi[totSize*TimeStep+arrayPos] >> phi[totSize*(nts+TimeStep)+arrayPos] >> theta[totSize*TimeStep+arrayPos] >> theta[totSize*(nts+TimeStep)+arrayPos] 
	    				   >> theta[totSize*(nts*2+TimeStep)+arrayPos];

	    			} else if(i<dataEnd){ // The rest of the data

	    				long long int arrayPos = i+frontHaloSize; // Shift across to account for the front halo at the start

	    				ic >> phi[totSize*TimeStep+arrayPos] >> phi[totSize*(nts+TimeStep)+arrayPos] >> theta[totSize*TimeStep+arrayPos] >> theta[totSize*(nts+TimeStep)+arrayPos] 
	    				   >> theta[totSize*(nts*2+TimeStep)+arrayPos];

	    			} else{ ic >> wasteData[0] >> wasteData[1] >> wasteData[2] >> wasteData[3] >> wasteData[4]; } // Don't need these so waste them into an unused variable 


	    		} else if(dataEnd>nPos){ // Need to deal with periodicity at the end of the data array

	    			if(i>=dataStart){ // All of the array except for the back halo

	    				long long int arrayPos = i-dataStart;

						ic >> phi[totSize*TimeStep+arrayPos] >> phi[totSize*(nts+TimeStep)+arrayPos] >> theta[totSize*TimeStep+arrayPos] >> theta[totSize*(nts+TimeStep)+arrayPos] 
	    				   >> theta[totSize*(nts*2+TimeStep)+arrayPos];				

	    			} else if(i<dataEnd%nPos){ // The back halo

	    				long long int arrayPos = i+coreSize+frontHaloSize;

	    				ic >> phi[totSize*TimeStep+arrayPos] >> phi[totSize*(nts+TimeStep)+arrayPos] >> theta[totSize*TimeStep+arrayPos] >> theta[totSize*(nts+TimeStep)+arrayPos] 
	    				   >> theta[totSize*(nts*2+TimeStep)+arrayPos];

	    			} else{ ic >> wasteData[0] >> wasteData[1] >> wasteData[2] >> wasteData[3] >> wasteData[4]; }

	    		} else{ // In the middle of the array so don't need to deal with periodicity

	    			if(i>=dataStart and i<dataEnd){

	    				long long int arrayPos = i-dataStart;

	    				ic >> phi[totSize*TimeStep+arrayPos] >> phi[totSize*(nts+TimeStep)+arrayPos] >> theta[totSize*TimeStep+arrayPos] >> theta[totSize*(nts+TimeStep)+arrayPos] 
	    				   >> theta[totSize*(nts*2+TimeStep)+arrayPos];

	    			} else{ ic >> wasteData[0] >> wasteData[1] >> wasteData[2] >> wasteData[3] >> wasteData[4]; }

	    		}

    		}
    	}

    } else if(ic_type=="random"){

    	// Use the seed to generate the data
		mt19937 generator (seed);
        uniform_real_distribution<double> distribution (0.0, 2*pi); // Uniform distribution for the phase of the strings
        double phase;

        // Skip the random numbers ahead to the appropriate point.
        for(i=0;i<coreStart;i++){ phase = distribution(generator); }



        for(i=frontHaloSize;i<coreSize+frontHaloSize;i++){

        	phase = distribution(generator);

        	phi[i] = eta*cos(phase);
        	phi[totSize*nts+i] = eta*sin(phase);

        	// Set next timestep as equal to the first
        	phi[totSize+i] = phi[i];
        	phi[totSize*(nts+1)+i] = phi[totSize*nts+i];

        	// Leave the gauge fields as zero (set by initialisation)

        }

        //cout << "Rank " << rank << "has phi[haloSize] = " << phi[haloSize] << ", and the next random number would be " << distribution(generator) << endl;

        // Now that the core data has been generated, need to communicate the haloes between processes. 

        for(comp=0;comp<2;comp++){ 

        	MPI_Sendrecv(&phi[totSize*nts*comp+frontHaloSize],nbrBackHaloSize,MPI_DOUBLE,(rank-1+size)%size,0, // Send this
        				 &phi[totSize*nts*comp+coreSize+frontHaloSize],backHaloSize,MPI_DOUBLE,(rank+1)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); // Receive this

        	MPI_Sendrecv(&phi[totSize*nts*comp+coreSize+frontHaloSize-nbrFrontHaloSize],nbrFrontHaloSize,MPI_DOUBLE,(rank+1)%size,0,
        				 &phi[totSize*nts*comp],frontHaloSize,MPI_DOUBLE,(rank-1+size)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        	MPI_Sendrecv(&phi[totSize*(nts*comp+1)+frontHaloSize],nbrBackHaloSize,MPI_DOUBLE,(rank-1+size)%size,1,
        				 &phi[totSize*(nts*comp+1)+coreSize+frontHaloSize],backHaloSize,MPI_DOUBLE,(rank+1)%size,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        	MPI_Sendrecv(&phi[totSize*(nts*comp+1)+coreSize+frontHaloSize-nbrFrontHaloSize],nbrFrontHaloSize,MPI_DOUBLE,(rank+1)%size,1,
        				 &phi[totSize*(nts*comp+1)],frontHaloSize,MPI_DOUBLE,(rank-1+size)%size,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        }

        // for(comp=0;comp<3;comp++){ 

        // 	MPI_Sendrecv(&theta[totSize*(nts*comp+tPast)+haloSize],haloSize,MPI_DOUBLE,(rank-1+size)%size,0,
        // 				 &theta[totSize*(nts*comp+tPast)+coreSize+haloSize],haloSize,MPI_DOUBLE,(rank+1)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        // 	MPI_Sendrecv(&theta[totSize*(nts*comp+tPast)+coreSize],haloSize,MPI_DOUBLE,(rank+1)%size,0,
        // 				 &theta[totSize*(nts*comp+tPast)],haloSize,MPI_DOUBLE,(rank-1+size)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        // }   	

    }

    gettimeofday(&end,NULL);

    if(rank==0){ cout << "Initial data loaded/generated in: " << end.tv_sec - start.tv_sec << "s" << endl; }

    gifStringPosFrame = 0;
    counter = 0;
    
	
    for(TimeStep=0;TimeStep<nt;TimeStep++){
		
		// string energyPoints = dir_path + "/DS/Energy_at_each_point_" + to_string(nx) + "_seed" + to_string(seed) + "the_iteration" + to_string(TimeStep) + ".txt";
		// ofstream energyData (energyPoints.c_str());

        double fric,tau;

        if(TimeStep>counter and rank==0){

            cout << "\rTimestep " << TimeStep-1 << " completed." << flush;

            counter += countRate;

        }

        if(expandDamp){ tau = 1 + (ntHeld+TimeStep)*dt; }
		else if(TimeStep < damped_nt){tau = 1 + (ntHeld)*dt;}
        else{ tau = 1 + (ntHeld+TimeStep-damped_nt)*dt; }

        // Is damping switched on or not?
        if(TimeStep<damped_nt){

            if(expandDamp){ fric = dampFac + alpha*scaling/tau; } // denominator is conformal time
            else{ fric = dampFac; }

        } else{

            if(expandDamp){ fric = alpha*scaling/tau; } // Time needs to have moved along during the damped phase
            else{ fric = alpha*scaling/tau; } // Time was not progressing during the damped phase

        }

        tNow = (TimeStep+1)%2;
        tPast = TimeStep%2;

        localEnergy = 0;
        deviationParameter = 0;
        vector<double> dens_en0 ,dens_en1, dens_en2, dens_en3, dens_en4, dens_en5, dens_en6, dens_en7, dens_en8, dens_en9;
        vector<double> xString, yString, zString; // Declares the vectors and clears them at every loop
		vector<double> axion(coreSize, 0.0);
		vector<double> phi1_out(coreSize, 0.0);
		vector<double> phi2_out(coreSize, 0.0);
		long double ro_a_sum = 0.0;
		vector<double> local_Energy(coreSize, 0.0);
		vector<double> chi_array(coreSize, 0.0);
		vector<double> chi_array_past(coreSize, 0.0);
		vector<double> spectr_phase_array(coreSize, 0.0);
		vector<double> spectr_phase_array_past(coreSize, 0.0);
        
		///////////////////////////////////////////// Absorbing BC starts //////////////////////////////////////////////////////////////
		
		if (xBC == "absorbing" && yBC == "absorbing" && zBC == "absorbing"){
			
			long long int Ind, ypInd, ymInd, zpInd, zmInd, xpInd, xmInd;

			// 6 faces of the cube
			for(j=1; j<ny-1; j++) { // 2 for x
				for(k=1; k<nz-1; k++){
					
					// x<0 boundary
					Ind = j*nz + k - dataStart;
					ypInd = (j+1)*nz + k - dataStart;
					ymInd = (j-1)*nz + k - dataStart;

					zpInd = j*nz + (k+1) - dataStart;
					zmInd = j*nz + (k-1) - dataStart;

					xpInd = (ny + j)*nz + k - dataStart;
					xmInd = j*nz + k - dataStart;	

					if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){

							// cout << "done" << endl;

							phiyy = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize * (nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								   + cos(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*comp + tNow) + ymInd] - c[comp] * sin(theta[totSize * (nts*1 + tNow) + ymInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ymInd]) / (dy * dy);

							phizz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
							     + cos(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*comp + tNow) + zmInd] - c[comp] * sin(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zmInd]) / (dz * dz);

							phitx = (cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - phi[totSize*(nts*comp + tNow) + xmInd]
								- cos(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd] + phi[totSize*(nts*comp + tPast) + xmInd]) / (dt * dx);

							phitt[comp] = phitx + 0.5*(phiyy + phizz);
							phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
						}
					}
					
					// x>0 boundary
					Ind = ((nx-1)*ny + j)*nz + k - dataStart;
					ypInd = ((nx-1)*ny + j+1)*nz + k - dataStart;
					ymInd = ((nx-1)*ny + j-1)*nz + k - dataStart;

					zpInd = ((nx-1)*ny + j)*nz + (k+1) - dataStart;
					zmInd = ((nx-1)*ny + j)*nz + (k-1) - dataStart;

					xmInd = ((nx-1)*ny + j)*nz + k - dataStart;
					xpInd = ((nx-2)*ny + j)*nz + k - dataStart;

					if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){
							
							
							phiyy = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
									+ cos(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*comp + tNow) + ymInd] - c[comp] * sin(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ymInd]) / (dy * dy);
							

							phizz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
									+ cos(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*comp + tNow) + zmInd] - c[comp] * sin(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zmInd]) / (dz * dz);
							
							
							phitx = (phi[totSize*(nts*comp + tNow) + xmInd] - cos(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd]
									- phi[totSize*(nts*comp + tPast) + xmInd] - cos(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd]) / (dt * dx);
						
							
							phitt[comp] = -phitx + 0.5*(phiyy + phizz);
							phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					
						}
					}
				}
			}
		

			for(i=1; i<nx-1; i++) { // 2 for y
				for(k=1; k<nz-1; k++){
					
					Ind = i*ny*nz + k - dataStart;

					ypInd = (i*ny + 1)*nz + k - dataStart;
					ymInd = (i*ny)*nz + k - dataStart;

					zpInd = (i*ny*nz + k + 1) - dataStart;
					zmInd = (i*ny*nz + k - 1) - dataStart;

					xpInd = ((i+1)*ny)*nz + k - dataStart;
					xmInd = ((i-1)*ny)*nz + k - dataStart;

					if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){

							// y<0 boundary

							phixx = (cos(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind] 
									+ cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xmInd] - c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xmInd]) / (dx * dx);

							phizz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
									+ cos(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*comp + tNow) + zmInd] - c[comp] * sin(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zmInd]) / (dz * dz);

							phity = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - phi[totSize*(nts*comp + tNow) + ymInd] 
									- cos(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd] + phi[totSize*(nts*comp + tPast) + ymInd]) / (dt * dy);

							phitt[comp] = phity + 0.5*(phixx + phizz);
							phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
						}
					}
					
					// y>0 boundary
					Ind = (i*ny + ny-1)*nz + k - dataStart;

					ymInd = (i*ny + ny-1)*nz + k - dataStart;
					ypInd = (i*ny + ny-2)*nz + k - dataStart;

					zpInd = (i*ny + ny-1)*nz + k + 1 - dataStart;
					zmInd = (i*ny + ny-1)*nz + k - 1 - dataStart;

					xpInd = ((i+1)*ny + ny-1)*nz + k - dataStart;
					xmInd = ((i-1)*ny + ny-1)*nz + k - dataStart;

					if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){

							phixx = (cos(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind] 
									+ cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xmInd] - c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xmInd]) / (dx * dx);

							phizz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
									+ cos(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*comp + tNow) + zmInd] - c[comp] * sin(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zmInd]) / (dz * dz);

							phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
									- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt * dy);
							
							phitt[comp] = -phity + 0.5*(phixx + phizz);
							phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
						}
					}
				}

			}

			for(i=1; i<nx-1; i++) { // 2 for z
				for(j=1; j<ny-1; j++) { 
				
					// z<0 boundary

					Ind = (i*ny + j)*nz - dataStart;

					ypInd = (i*ny + j + 1)*nz - dataStart;
					ymInd = (i*ny + j - 1)*nz - dataStart;

					zpInd = (i*ny + j)*nz + 1 - dataStart;
					zmInd = (i*ny + j)*nz - dataStart;

					xpInd = ((i+1)*ny + j)*nz - dataStart;
					xmInd = ((i-1)*ny + j)*nz - dataStart;

					if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){

							phixx = (cos(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind] 
									+ cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xmInd] - c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xmInd]) / (dx * dx);

							phiyy = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
									+ cos(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*comp + tNow) + ymInd] - c[comp] * sin(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ymInd]) / (dy * dy);

							phitz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - phi[totSize*(nts*comp + tNow) + zmInd]
									- cos(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd] + phi[totSize*(nts*comp + tPast) + zmInd]) / (dt * dy);

							phitt[comp] = phitz + 0.5*(phixx + phiyy);
							phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
						}
					}

					// z>0 boundary
					Ind = (i*ny + j)*nz + nz-1 - dataStart;

					ypInd = (i*ny + j + 1)*nz + nz-1 - dataStart;
					ymInd = (i*ny + j - 1)*nz + nz-1 - dataStart;

					zpInd = (i*ny + j)*nz + nz-1 - dataStart;
					zmInd = (i*ny + j)*nz + nz-2 - dataStart;

					xpInd = ((i+1)*ny + j)*nz + nz-1 - dataStart;
					xmInd = ((i-1)*ny + j)*nz + nz-1 - dataStart;

					if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){

							phixx = (cos(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind] 
									+ cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xmInd] - c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xmInd]) / (dx * dx);

							phiyy = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
									+ cos(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*comp + tNow) + ymInd] - c[comp] * sin(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ymInd]) / (dy * dy);

							phitz = (phi[totSize*(nts*comp + tNow) + zmInd] - cos(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd]
									- phi[totSize*(nts*comp + tPast) + zmInd] + cos(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd]) / (dt * dy);
								
							phitt[comp] = -phitz + 0.5*(phixx + phiyy);
							phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];				
						}
					}
				}
			}	
		
			cout << "faces done " << endl;
			// 12 edges of the cube
			for(k=1; k<nz-1; k++) { // 4 for z

				// x,y<0 edge

				Ind = k - dataStart;

				ypInd = nz + k - dataStart;
				ymInd = k - dataStart;

				zpInd = k+1 - dataStart;
				zmInd = k-1 - dataStart;

				xpInd = ny*nz + k - dataStart;
				xmInd = k - dataStart;
				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phizz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*comp + tNow) + zmInd] - c[comp] * sin(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zmInd]) / (dz * dz);

						phitx = (cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - phi[totSize*(nts*comp + tNow) + xmInd]
								- cos(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd] + phi[totSize*(nts*comp + tPast) + xmInd]) / (dt * dx);

						phity = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - phi[totSize*(nts*comp + tNow) + ymInd]
								- cos(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd] + phi[totSize*(nts*comp + tPast) + ymInd]) / (dt * dy);

						phitt[comp] = ( phizz + 2*(phitx + phity) )/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}

				// x<0,y>0 edge

				Ind = (ny-1)*nz + k - dataStart;

				ypInd = (ny-2)*nz + k - dataStart;
				ymInd = (ny-1)*nz + k - dataStart;

				zpInd = (ny-1)*nz + k+1 - dataStart;
				zmInd = (ny-1)*nz + k-1 - dataStart;

				xpInd = (ny + ny-1)*nz + k - dataStart;
				xmInd = (ny-1)*nz + k - dataStart;
				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phizz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*comp + tNow) + zmInd] - c[comp] * sin(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zmInd]) / (dz * dz);

						phitx = (cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - phi[totSize*(nts*comp + tNow) + xmInd]
								- cos(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd] + phi[totSize*(nts*comp + tPast) + xmInd]) / (dt * dx);

						phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
								- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt * dy);

						phitt[comp] = ( phizz + 2*(phitx - phity) )/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}

				// x>0,y<0 edge

				Ind = (nx-1)*ny*nz + k - dataStart;

				ypInd = ((nx-1)*ny + 1)*nz + k - dataStart;
				ymInd = ((nx-1)*ny)*nz + k - dataStart;

				zpInd = (nx-1)*ny*nz + k+1 - dataStart;
				zmInd = (nx-1)*ny*nz + k-1 - dataStart;

				xpInd = (nx-2)*ny*nz + k - dataStart;
				xmInd = (nx-1)*ny*nz + k - dataStart;

				if (coreStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phizz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*comp + tNow) + zmInd] - c[comp] * sin(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zmInd]) / (dz * dz);

						phitx = (phi[totSize*(nts*comp + tNow) + xmInd] - cos(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd]
								- phi[totSize*(nts*comp + tPast) + xmInd] - cos(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd]) / (dt * dx);

						phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
								- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt * dy);

						phitt[comp] = ( phizz - 2*(phitx - phity) )/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}
				// x,y>0 edge

				Ind = ((nx-1)*ny + ny-1)*nz + k - dataStart;

				ypInd = ((nx-1)*ny + ny-2)*nz + k - dataStart;
				ymInd = ((nx-1)*ny + ny-1)*nz + k - dataStart;

				zpInd = ((nx-1)*ny + ny-1)*nz + k+1 - dataStart;
				zmInd = ((nx-1)*ny + ny-1)*nz + k-1 - dataStart;

				xpInd = ((nx-2)*ny + ny-1)*nz + k - dataStart;
				xmInd = ((nx-1)*ny + ny-1)*nz + k - dataStart;

				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phizz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*comp + tNow) + zmInd] - c[comp] * sin(theta[totSize*(nts*2 + tNow) + zmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zmInd]) / (dz * dz);

						phitx = (phi[totSize*(nts*comp + tNow) + xmInd] - cos(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd]
								- phi[totSize*(nts*comp + tPast) + xmInd] - cos(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd]) / (dt * dx);

						phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
								- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt * dy);

						phitt[comp] = ( phizz - 2*(phitx + phity) )/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}
			}	
			for(i=1; i<nx-1; i++) { // 4 for x
				
				// y,z<0 edge

				Ind = i*ny*nz - dataStart;

				ypInd = (i*ny + 1)*nz - dataStart;
				ymInd = (i*ny)*nz - dataStart;

				zpInd = i*ny*nz + 1 - dataStart;
				zmInd = i*ny*nz - dataStart;

				xpInd = (i+1)*ny*nz - dataStart;
				xmInd = i*ny*nz - dataStart;
				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){
					
						phixx = (cos(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xmInd] - c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xmInd]) / (dx * dx);

						phity = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - phi[totSize*(nts*comp + tNow) + ymInd]
								- cos(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd] + phi[totSize*(nts*comp + tPast) + ymInd]) / (dt * dy);

						phitz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - phi[totSize*(nts*comp + tNow) + zmInd]
								- cos(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd] + phi[totSize*(nts*comp + tPast) + zmInd]) / (dt * dy);

						phitt[comp] = (phixx + 2*(phity + phitz))/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}
				// y,z>0 edge
				
				Ind = (i*ny + ny-1)*nz + nz-1 - dataStart;

				ypInd = (i*ny + ny-2)*nz + nz-1 - dataStart;
				ymInd = (i*ny + ny-1)*nz + nz-1 - dataStart;

				zpInd = (i*ny + ny-1)*nz + nz-2 - dataStart;
				zmInd = (i*ny + ny-1)*nz + nz-1 - dataStart;

				xpInd = ((i+1)*ny + ny-1)*nz + nz-1 - dataStart;
				xmInd = (i*ny + ny-1)*nz + nz-1 - dataStart;

				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){
							
						phixx = (cos(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xmInd] - c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xmInd]) / (dx * dx);

						phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
								- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt * dy);

						phitz = (phi[totSize*(nts*comp + tNow) + zmInd] - cos(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd]
								- phi[totSize*(nts*comp + tPast) + zmInd] + cos(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd]) / (dt * dy);

						phitt[comp] = (phixx - 2*(phity + phitz))/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}

				// y>0 z<0 edge

				Ind = (i*ny + ny-1)*nz - dataStart;

				ypInd = (i*ny + ny-2)*nz - dataStart;
				ymInd = (i*ny + ny-1)*nz - dataStart;

				zpInd = (i*ny + ny-1)*nz + 1 - dataStart;
				zmInd = (i*ny + ny-1)*nz - dataStart;

				xpInd = ((i+1)*ny + ny-1)*nz - dataStart;
				xmInd = (i*ny + ny-1)*nz - dataStart;

				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phixx = (cos(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xmInd] - c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xmInd]) / (dx * dx);

						phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
								- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt * dy);

						phitz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - phi[totSize*(nts*comp + tNow) + zmInd]
								- cos(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd] + phi[totSize*(nts*comp + tPast) + zmInd]) / (dt * dy);

						phitt[comp] = (phixx + 2*(- phity + phitz))/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}

				// y<0 z>0 edge

				Ind = i*ny*nz + nz-1 - dataStart;

				ypInd = (i*ny + 1)*nz + nz-1 - dataStart;
				ymInd = (i*ny)*nz + nz-1 - dataStart;

				zpInd = i*ny*nz + nz-2 - dataStart;
				zmInd = i*ny*nz + nz-1 - dataStart;

				xpInd = (i+1)*ny*nz + nz-1 - dataStart;
				xmInd = i*ny*nz + nz-1 - dataStart;

				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phixx = (cos(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind] 
								+ cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xmInd] - c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xmInd]) / (dx * dx);

						phity = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - phi[totSize*(nts*comp + tNow) + ymInd] 
								- cos(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd] + phi[totSize*(nts*comp + tPast) + ymInd]) / (dt * dy);

						phitz = (phi[totSize*(nts*comp + tNow) + zmInd] - cos(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd]
								- phi[totSize*(nts*comp + tPast) + zmInd] + cos(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd]) / (dt * dy);

						phitt[comp] = (phixx + 2*(phity - phitz))/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}
			}
			for(j=1; j<ny-1; j++) { // 4 for y

				// x,z<0 edge

				Ind = j*nz - dataStart;

				ypInd = (j+1)*nz - dataStart;
				ymInd = (j-1)*nz - dataStart;

				zpInd = j*nz + 1 - dataStart;
				zmInd = j*nz - dataStart;

				xpInd = (ny + j)*nz - dataStart;
				xmInd = j*nz - dataStart;
				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phiyy = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize * (nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*comp + tNow) + ymInd] - c[comp] * sin(theta[totSize * (nts*1 + tNow) + ymInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ymInd]) / (dy * dy);

						phitx = (cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - phi[totSize*(nts*comp + tNow) + xmInd]
								- cos(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd] + phi[totSize*(nts*comp + tPast) + xmInd]) / (dt * dx);

						phitz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - phi[totSize*(nts*comp + tNow) + zmInd]
								- cos(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd] + phi[totSize*(nts*comp + tPast) + zmInd]) / (dt * dy);

						phitt[comp] = ( phiyy + 2*(phitx + phitz) )/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}

				// x,z>0 edge

				Ind = ((nx-1)*ny + j)*nz + nz-1 - dataStart;

				ypInd = ((nx-1)*ny + j+1)*nz + nz-1 - dataStart;
				ymInd = ((nx-1)*ny + j-1)*nz + nz-1 - dataStart;

				zpInd = ((nx-1)*ny + j)*nz + nz-2 - dataStart;
				zmInd = ((nx-1)*ny + j)*nz + nz-1 - dataStart;

				xpInd = ((nx-2)*ny + j)*nz + nz-1 - dataStart;
				xmInd = ((nx-1)*ny + j)*nz + nz-1 - dataStart;

				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phiyy = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize * (nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*comp + tNow) + ymInd] - c[comp] * sin(theta[totSize * (nts*1 + tNow) + ymInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ymInd]) / (dy * dy);

						phitx = (phi[totSize*(nts*comp + tNow) + xmInd] - cos(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd]
								- phi[totSize*(nts*comp + tPast) + xmInd] - cos(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd]) / (dt * dx);

						phitz = (phi[totSize*(nts*comp + tNow) + zmInd] - cos(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd]
								- phi[totSize*(nts*comp + tPast) + zmInd] + cos(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd]) / (dt * dy);

						phitt[comp] = (phiyy - 2*(phitx + phitz))/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}

				// x>0,z<0 edge

				Ind = ((nx-1)*ny + j)*nz - dataStart;

				ypInd = ((nx-1)*ny + j+1)*nz - dataStart;
				ymInd = ((nx-1)*ny + j-1)*nz - dataStart;

				zpInd = ((nx-1)*ny + j)*nz + 1 - dataStart;
				zmInd = ((nx-1)*ny + j)*nz - dataStart;

				xpInd = ((nx-2)*ny + j)*nz - dataStart;
				xmInd = ((nx-1)*ny + j)*nz - dataStart;
				
				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phiyy = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize * (nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*comp + tNow) + ymInd] - c[comp] * sin(theta[totSize * (nts*1 + tNow) + ymInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ymInd]) / (dy * dy);

						phitx = (phi[totSize*(nts*comp + tNow) + xmInd] - cos(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd]
								- phi[totSize*(nts*comp + tPast) + xmInd] - cos(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd]) / (dt * dx);

						phitz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - phi[totSize*(nts*comp + tNow) + zmInd]
								- cos(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd] + phi[totSize*(nts*comp + tPast) + zmInd]) / (dt * dy);

						phitt[comp] = (phiyy + 2*(- phitx + phitz))/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}

				// x<0,z>0 edge

				Ind = j*nz + nz-1 - dataStart;

				ypInd = (j+1)*nz + nz-1 - dataStart;
				ymInd = (j-1)*nz + nz-1 - dataStart;

				zpInd = j*nz + nz-2 - dataStart;
				zmInd = j*nz + nz-1 - dataStart;

				xpInd = ny*nz + j*nz + nz-1 - dataStart;
				xmInd = j*nz + nz-1 - dataStart;
				if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
					for(comp=0;comp<2;comp++){

						phiyy = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize * (nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - 2 * phi[totSize*(nts*comp + tNow) + Ind]
								+ cos(theta[totSize*(nts*1 + tNow) + ymInd]) * phi[totSize*(nts*comp + tNow) + ymInd] - c[comp] * sin(theta[totSize * (nts*1 + tNow) + ymInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ymInd]) / (dy * dy);

						phitx = (cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - phi[totSize*(nts*comp + tNow) + xmInd]
								- cos(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd] + phi[totSize*(nts*comp + tPast) + xmInd]) / (dt * dx);

						phitz = (phi[totSize*(nts*comp + tNow) + zmInd] - cos(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd]
								- phi[totSize*(nts*comp + tPast) + zmInd] + cos(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd]) / (dt * dy);

						phitt[comp] = ( phiyy + 2*(phitx - phitz) )/3;
						phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
					}
				}
			}

			cout << "edges done" << endl;
			// 8 corners of the cube

			// x,y,z<0 corner

			Ind = 0 - dataStart;

			ypInd = nz - dataStart;
			ymInd = 0 - dataStart;

			zpInd = 1 - dataStart;
			zmInd = 0 - dataStart;

			xpInd = ny*nz - dataStart;
			xmInd = 0 - dataStart;
			
			if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
				for(comp=0;comp<2;comp++){

					phitx = (cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - phi[totSize*(nts*comp + tNow) + xmInd]
							- cos(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd] + phi[totSize*(nts*comp + tPast) + xmInd]) / (dt * dx);
							
					phity = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - phi[totSize*(nts*comp + tNow) + ymInd]
							- cos(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd] + phi[totSize*(nts*comp + tPast) + ymInd]) / (dt * dy);

					phitz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - phi[totSize*(nts*comp + tNow) + zmInd]
							- cos(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd] + phi[totSize*(nts*comp + tPast) + zmInd]) / (dt * dy);

					phitt[comp] = ( phitx + phity + phitz )/2;
					phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
				}
			}
			// x,y,z>0 corner

			Ind = ((nx-1)*ny + ny-1)*nz + nz-1 - dataStart;

			ypInd = ((nx-1)*ny + ny-2)*nz + nz-1 - dataStart;
			ymInd = ((nx-1)*ny + ny-1)*nz + nz-1 - dataStart;

			zpInd = ((nx-1)*ny + ny-1)*nz + nz-2 - dataStart;
			zmInd = ((nx-1)*ny + ny-1)*nz + nz-1 - dataStart;

			xpInd = ((nx-2)*ny + ny-1)*nz + nz-1 - dataStart;
			xmInd = ((nx-1)*ny + ny-1)*nz + nz-1 - dataStart;

			if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
				for(comp=0;comp<2;comp++){

					phitx = (phi[totSize*(nts*comp + tNow) + xmInd] - cos(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd]
							- phi[totSize*(nts*comp + tPast) + xmInd] - cos(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd]) / (dt * dx);

					phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
							- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt * dy);

					phitz = (phi[totSize*(nts*comp + tNow) + zmInd] - cos(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd]
							- phi[totSize*(nts*comp + tPast) + zmInd] + cos(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd]) / (dt * dy);

					phitt[comp] = - ( phitx + phity + phitz )/2;
					phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
				}
			}

			// x<0 y,z>0 corner

			Ind = (ny-1)*nz + nz-1 - dataStart;

			ypInd = (ny-2)*nz + nz-1 - dataStart;
			ymInd = (ny-1)*nz + nz-1 - dataStart;

			zpInd = (ny-1)*nz + nz-2 - dataStart;
			zmInd = (ny-1)*nz + nz-1 - dataStart;

			xpInd = ny*nz + (ny-1)*nz + nz-1 - dataStart;
			xmInd = (ny-1)*nz + nz-1 - dataStart;

			if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
				for(comp=0;comp<2;comp++){

					phitx = (cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - phi[totSize*(nts*comp + tNow) + xmInd]
							- cos(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd] + phi[totSize*(nts*comp + tPast) + xmInd]) / (dt * dx);

					phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
							- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt * dy);

					phitz = (phi[totSize*(nts*comp + tNow) + zmInd] - cos(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd]
							- phi[totSize*(nts*comp + tPast) + zmInd] + cos(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd]) / (dt * dy);

					phitt[comp] = ( phitx - phity - phitz )/2;
					phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];

				}
			}

			// x>0 y,z<0 corner

			Ind = ((nx-1)*ny)*nz - dataStart;

			ypInd = ((nx-1)*ny + 1)*nz - dataStart;
			ymInd = ((nx-1)*ny)*nz - dataStart;

			zpInd = ((nx-1)*ny)*nz + 1 - dataStart;
			zmInd = ((nx-1)*ny)*nz - dataStart;

			xpInd = ((nx-2)*ny)*nz - dataStart;
			xmInd = ((nx-1)*ny)*nz - dataStart;

			if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
				for(comp=0;comp<2;comp++){

					phitx = (phi[totSize*(nts*comp + tNow) + xmInd] - cos(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd]
							- phi[totSize*(nts*comp + tPast) + xmInd] - cos(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xpInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd]) / (dt * dx);

					phity = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - phi[totSize*(nts*comp + tNow) + ymInd]
							- cos(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd] + phi[totSize*(nts*comp + tPast) + ymInd]) / (dt * dy);

					phitz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - phi[totSize*(nts*comp + tNow) + zmInd]
							- cos(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd] + phi[totSize*(nts*comp + tPast) + zmInd]) / (dt * dy);

					phitt[comp] = (- phitx + phity + phitz )/2;
					phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
				}
			}
			// x,y<0 z>0 corner

			Ind = nz-1 - dataStart;

			ypInd = nz + nz-1 - dataStart;
			ymInd = 0 + nz-1 - dataStart;

			zpInd = nz-2 - dataStart;
			zmInd = nz-1 - dataStart;

			xpInd = ny*nz + nz-1 - dataStart;
			xmInd = nz-1 - dataStart;

			if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
				for(comp=0;comp<2;comp++){
				
					phitx = (cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp + c[comp]) + tNow) + xpInd] - phi[totSize*(nts*comp + tNow) + xmInd] 
							- cos(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*(comp + c[comp]) + tPast) + xpInd] + phi[totSize*(nts*comp + tPast) + xmInd]) / (dt * dx);

					phity = (cos(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + Ind]) * phi[totSize*(nts*(comp + c[comp]) + tNow) + ypInd] - phi[totSize*(nts*comp + tNow) + ymInd] 
							- cos(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + Ind]) * phi[totSize*(nts*(comp + c[comp]) + tPast) + ypInd] + phi[totSize*(nts*comp + tPast) + ymInd]) / (dt * dy);

					phitz = (phi[totSize*(nts*comp + tNow) + zmInd] - cos(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + zpInd]) * phi[totSize*(nts*(comp + c[comp]) + tNow) + zpInd] 
							- phi[totSize*(nts*comp + tPast) + zmInd] + cos(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + zpInd]) * phi[totSize*(nts*(comp + c[comp]) + tPast) + zpInd]) / (dt * dy);

					phitt[comp] = ( phitx + phity - phitz )/2;
					phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
				}
			}

			// x>0 y<0 z>0 corner

			Ind = (nx-1)*ny*nz + nz-1 - dataStart;

			ypInd = ((nx-1)*ny + 1)*nz + nz-1 - dataStart;
			ymInd = ((nx-1)*ny)*nz + nz-1 - dataStart;

			zpInd = (nx-1)*ny*nz + nz-2 - dataStart;
			zmInd = (nx-1)*ny*nz + nz-1 - dataStart;

			xpInd = ((nx-2)*ny)*nz + nz-1 - dataStart;
			xmInd = (nx-1)*ny*nz + nz-1 - dataStart;

			if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
				for(comp=0;comp<2;comp++){

					phitx = (phi[totSize*(nts*comp + tNow) + xmInd] - cos(theta[totSize*(nts*0 + tNow) + xpInd])*phi[totSize*(nts*comp + tNow) + xpInd] + c[comp]*sin(theta[totSize*(nts*0 + tNow) + xpInd])*phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] 
							- phi[totSize*(nts*comp + tPast) + xmInd] - cos(theta[totSize*(nts*0 + tPast) + xpInd])*phi[totSize*(nts*comp + tPast) + xpInd] - c[comp]*sin(theta[totSize*(nts*0 + tPast) + xpInd])*phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd]) / (dt*dx);

					phity = (cos(theta[totSize*(nts*1 + tNow) + Ind])*phi[totSize*(nts*comp + tNow) + ypInd] + c[comp]*sin(theta[totSize*(nts*1 + tNow) + Ind])*phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] - phi[totSize*(nts*comp + tNow) + ymInd] 
							- cos(theta[totSize*(nts*1 + tPast) + Ind])*phi[totSize*(nts*comp + tPast) + ypInd] - c[comp]*sin(theta[totSize*(nts*1 + tPast) + Ind])*phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd] + phi[totSize*(nts*comp + tPast) + ymInd]) / (dt*dy);

					phitz = (phi[totSize*(nts*comp + tNow) + zmInd] - cos(theta[totSize*(nts*2 + tNow) + zpInd])*phi[totSize*(nts*comp + tNow) + zpInd] + c[comp]*sin(theta[totSize*(nts*2 + tNow) + zpInd])*phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] 
							- phi[totSize*(nts*comp + tPast) + zmInd] + cos(theta[totSize*(nts*2 + tPast) + zpInd])*phi[totSize*(nts*comp + tPast) + zpInd] - c[comp]*sin(theta[totSize*(nts*2 + tPast) + zpInd])*phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd]) / (dt*dy);

					phitt[comp] = (- phitx + phity - phitz )/2;
					phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
				}
			}

			// x>0 y>0 z<0 corner

			Ind = ((nx-1)*ny + ny-1)*nz - dataStart;

			ypInd = ((nx-1)*ny + ny-2)*nz - dataStart;
			ymInd = ((nx-1)*ny + ny-1)*nz - dataStart;

			zpInd = ((nx-1)*ny + ny-1)*nz + 1 - dataStart;
			zmInd = ((nx-1)*ny + ny-1)*nz - dataStart;

			xpInd = ((nx-2)*ny + ny-1)*nz - dataStart;
			xmInd = ((nx-1)*ny + ny-1)*nz - dataStart;

			if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
				for(comp=0;comp<2;comp++){

					phitx = (phi[totSize*(nts*comp + tNow) + xmInd] - cos(theta[totSize*(nts*0 + tNow) + xpInd])*phi[totSize*(nts*comp + tNow) + xpInd] + c[comp]*sin(theta[totSize*(nts*0 + tNow) + xpInd])*phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd]
							- phi[totSize*(nts*comp + tPast) + xmInd] - cos(theta[totSize*(nts*0 + tPast) + xpInd])*phi[totSize*(nts*comp + tPast) + xpInd] - c[comp]*sin(theta[totSize*(nts*0 + tPast) + xpInd])*phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd]) / (dt*dx);

					phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd])*phi[totSize*(nts*comp + tNow) + ypInd] + c[comp]*sin(theta[totSize*(nts*1 + tNow) + ypInd])*phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
							- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd])*phi[totSize*(nts*comp + tPast) + ypInd] - c[comp]*sin(theta[totSize*(nts*1 + tPast) + ypInd])*phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt*dy);

					phitz = (cos(theta[totSize*(nts*2 + tNow) + Ind])*phi[totSize*(nts*comp + tNow) + zpInd] + c[comp]*sin(theta[totSize*(nts*2 + tNow) + Ind])*phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - phi[totSize*(nts*comp + tNow) + zmInd]
							- cos(theta[totSize*(nts*2 + tPast) + Ind])*phi[totSize*(nts*comp + tPast) + zpInd] - c[comp]*sin(theta[totSize*(nts*2 + tPast) + Ind])*phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd] + phi[totSize*(nts*comp + tPast) + zmInd]) / (dt*dy);

					phitt[comp] = (- phitx - phity + phitz )/2;
					phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
				}
			}

			// x<0 y>0 z<0 corner

			Ind = (ny-1)*nz - dataStart;

			ypInd = (ny-2)*nz - dataStart;
			ymInd = (ny-1)*nz - dataStart;

			zpInd = (ny-1)*nz + 1 - dataStart;
			zmInd = (ny-1)*nz - dataStart;

			xpInd = ny*nz + (ny-1)*nz - dataStart;
			xmInd = (ny-1)*nz - dataStart;
			
			if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
				for(comp=0;comp<2;comp++){

					phitx = (cos(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd] - phi[totSize*(nts*comp + tNow) + xmInd] 
							- cos(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*comp + tPast) + xpInd] - c[comp] * sin(theta[totSize*(nts*0 + tPast) + xmInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + xpInd] + phi[totSize*(nts*comp + tPast) + xmInd]) / (dt*dx);

					phity = (phi[totSize*(nts*comp + tNow) + ymInd] - cos(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*comp + tNow) + ypInd] + c[comp] * sin(theta[totSize*(nts*1 + tNow) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd]
							- phi[totSize*(nts*comp + tPast) + ymInd] + cos(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*comp + tPast) + ypInd] - c[comp] * sin(theta[totSize*(nts*1 + tPast) + ypInd]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + ypInd]) / (dt*dy);
							
					phitz = (cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd] - phi[totSize*(nts*comp + tNow) + zmInd] 
							- cos(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*comp + tPast) + zpInd] - c[comp] * sin(theta[totSize*(nts*2 + tPast) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tPast) + zpInd] + phi[totSize*(nts*comp + tPast) + zmInd]) / (dt*dy);

					phitt[comp] = ( phitx - phity + phitz )/2;
					phi[totSize*(nts*comp+tPast)+Ind] = 2*phi[totSize*(nts*comp+tNow)+Ind] - phi[totSize*(nts*comp+tPast)+Ind] + dt*dt*phitt[comp];
				}
			}
		}

		//////////////////////////////////////////// Absorbing BC Ends /////////////////////////////////////////////////////////////////

		//////////////////////////////////////////// Neumann BC Starts /////////////////////////////////////////////////////////////////

		if (xBC == "neumann" && yBC == "neumann" && zBC == "neumann"){

			long long int Ind, ypInd, yppInd, ypppInd, xpInd, xppInd, xpppInd, zpInd, zppInd, zpppInd;

			for(i=1; i<nx-1; i++) {
				for(k=1; k<nz-1; k++){

					Ind = i*ny*nz + k - dataStart;
					ypInd = (i*ny + 1)*nz + k - dataStart;

					yppInd = (i*ny + ny-1)*nz + k - dataStart;
					ypppInd = (i*ny + ny-2)*nz + k - dataStart;

					if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){
							
							// cout << phi[totSize*(nts*comp + tNow) + Ind] << " " << phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd] << " " << theta[totSize*(nts*1 + tNow) + Ind] << endl;
							
							phi[totSize*(nts*comp + tNow) + Ind] = cos(theta[totSize*(nts*1 + tNow) + Ind])*phi[totSize*(nts*comp + tNow) + ypInd] + c[comp]*sin(theta[totSize*(nts*1 + tNow) + Ind])*phi[totSize*(nts*(comp+c[comp]) + tNow) + ypInd];
						}
					}

					if (coreStart - dataStart <= yppInd && yppInd < coreEnd - dataStart) {
						for(comp=0;comp<2;comp++){

							phi[totSize*(nts*comp + tNow) + yppInd] = cos(theta[totSize*(nts*1 + tNow) + ypppInd])*phi[totSize*(nts*comp + tNow) + ypppInd] - c[comp]*sin(theta[totSize*(nts*1 + tNow) + ypppInd])*phi[totSize*(nts*(comp+c[comp]) + tNow) + ypppInd];
						}
					}
				}
			}
		
			// cout << "Neumann BC for y done" << endl;
		
			for(j=0; j<ny; j++) { // 2 for x
				for(k=1; k<nz-1; k++){

					Ind = j*nz + k - dataStart;
					xpInd = (ny + j)*nz + k - dataStart;
					
					xppInd = ((nx-1)*ny + j)*nz + k - dataStart;
					xpppInd = ((nx-2)*ny + j)*nz + k - dataStart;

					if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){

							phi[totSize*(nts*comp + tNow) + Ind] = cos(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + xpInd] + c[comp] * sin(theta[totSize*(nts*0 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpInd];
						}
					}
					if (coreStart - dataStart <= xppInd && xppInd < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){

							phi[totSize*(nts*comp + tNow) + xppInd] = cos(theta[totSize*(nts*0 + tNow) + xpppInd]) * phi[totSize*(nts*comp + tNow) + xpppInd] - c[comp] * sin(theta[totSize*(nts*0 + tNow) + xpppInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + xpppInd];
						}
					}
				}
			}
			// cout << "Neumann BC for x done" << endl;
            for(i=0;i<nx;i++){
                for(j=0;j<ny;j++){

					Ind = (i*ny + j)*nz - dataStart;
					zpInd = (i*ny + j)*nz + 1 - dataStart;
					
					zppInd = (i*ny + j)*nz + nz-1 - dataStart;
					zpppInd = (i*ny + j)*nz + nz-2 - dataStart;

					if (coreStart - dataStart <= Ind && Ind < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){
							
							phi[totSize*(nts*comp + tNow) + Ind] = cos(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*comp + tNow) + zpInd] + c[comp] * sin(theta[totSize*(nts*2 + tNow) + Ind]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpInd];
						}
					}

					if (coreStart - dataStart <= zppInd && zppInd < coreEnd - dataStart){
						for(comp=0;comp<2;comp++){
							
							phi[totSize*(nts*comp + tNow) + zppInd] = cos(theta[totSize*(nts*2 + tNow) + zpppInd]) * phi[totSize*(nts*comp + tNow) + zpppInd] - c[comp] * sin(theta[totSize*(nts*2 + tNow) + zpppInd]) * phi[totSize*(nts*(comp+c[comp]) + tNow) + zpppInd];
						}
					}
				}
			}
			// cout << "Neumann BC for z done" << endl;
		}

		//////////////////////////////////////////// Neumann BC Ends ///////////////////////////////////////////////////////////////////	

		for(i=frontHaloSize;i<coreSize+frontHaloSize;i++){ // Now evolve the core data

			int xInd = (i + dataStart) / (ny * nz);
			int yInd = ((i + dataStart) / nz) % ny;
			int zInd = i % nz;

			if (((xBC == "fixed" || xBC == "absorbing" || xBC == "neumann") && (xInd == 0 || xInd == nx - 1)) ||
				((yBC == "fixed" || yBC == "absorbing" || yBC == "neumann") && (yInd == 0 || yInd == ny - 1)) ||
				((zBC == "fixed" || zBC == "absorbing" || zBC == "neumann") && (zInd == 0 || zInd == nz - 1))) {
				// Boundary condition is fixed or absorbing or neumann, skip evolution
				continue;
			}

			else{
				imx = i-ny*nz;
				ipx = i+ny*nz;

				imy = (i+dataStart-nz+ny*nz)%(ny*nz) + ( (i+dataStart)/(ny*nz) )*ny*nz - dataStart; 
				ipy = (i+dataStart+nz)%(ny*nz) + ( (i+dataStart)/(ny*nz) )*ny*nz - dataStart;

				imz = (i+dataStart-1+nz)%nz + ( (i+dataStart)/nz )*nz - dataStart;
				ipz = (i+dataStart+1)%nz + ( (i+dataStart)/nz )*nz - dataStart;

				ipxmy = imy+ny*nz;
				ipxmz = imz+ny*nz;
				imxpy = ipy-ny*nz;
				ipymz = (ipy+dataStart-1+nz)%nz + ( (ipy+dataStart)/nz )*nz - dataStart;
				imxpz = ipz-ny*nz;
				imypz = (imy+dataStart+1)%nz + ( (imy+dataStart)/nz )*nz - dataStart;


				phiMagSqr = pow(phi[totSize*tNow+i],2) + pow(phi[totSize*(nts+tNow)+i],2); 
				phiMagSqr_past = pow(phi[totSize*tPast+i],2) + pow(phi[totSize*(nts+tPast)+i],2); 

				// Loop over phi components

				for(comp=0;comp<2;comp++){

					// 2nd order spatial derivatives calculated with 2nd order finite difference

					// c is 1 when comp = 0 and -1 when comp = 1

					phixx = ( cos(theta[totSize*tNow+i])*phi[totSize*(nts*comp+tNow)+ipx] + c[comp]*sin(theta[totSize*tNow+i])*phi[totSize*(nts*(comp+c[comp])+tNow)+ipx] - 2*phi[totSize*(nts*comp+tNow)+i]
							+ cos(theta[totSize*tNow+imx])*phi[totSize*(nts*comp+tNow)+imx] - c[comp]*sin(theta[totSize*tNow+imx])*phi[totSize*(nts*(comp+c[comp])+tNow)+imx] )/(dx*dx);

					phiyy = ( cos(theta[totSize*(nts+tNow)+i])*phi[totSize*(nts*comp+tNow)+ipy] + c[comp]*sin(theta[totSize*(nts+tNow)+i])*phi[totSize*(nts*(comp+c[comp])+tNow)+ipy] - 2*phi[totSize*(nts*comp+tNow)+i]
							+ cos(theta[totSize*(nts+tNow)+imy])*phi[totSize*(nts*comp+tNow)+imy] - c[comp]*sin(theta[totSize*(nts+tNow)+imy])*phi[totSize*(nts*(comp+c[comp])+tNow)+imy] )/(dy*dy);

					phizz = ( cos(theta[totSize*(nts*2+tNow)+i])*phi[totSize*(nts*comp+tNow)+ipz] + c[comp]*sin(theta[totSize*(nts*2+tNow)+i])*phi[totSize*(nts*(comp+c[comp])+tNow)+ipz] - 2*phi[totSize*(nts*comp+tNow)+i]
							+ cos(theta[totSize*(nts*2+tNow)+imz])*phi[totSize*(nts*comp+tNow)+imz] - c[comp]*sin(theta[totSize*(nts*2+tNow)+imz])*phi[totSize*(nts*(comp+c[comp])+tNow)+imz] )/(dz*dz);


					phit[comp] = ( phi[totSize*(nts*comp+tNow)+i] - phi[totSize*(nts*comp+tPast)+i] )/dt;

					// Calculate the second order time derivative and update the field
					phitt[comp] = phixx + phiyy + phizz - pow( pow(tau,scaling), beta_phi)*( 0.5*lambda*(phiMagSqr - pow(eta,2))*phi[totSize*(nts*comp+tNow)+i] ) - fric*phit[comp];
					phi[totSize*(nts*comp+tPast)+i] = 2*phi[totSize*(nts*comp+tNow)+i] - phi[totSize*(nts*comp+tPast)+i] + dt*dt*phitt[comp];

				}


				curx = cos(theta[totSize*tNow+i])*(phi[totSize*(nts+tNow)+i]*phi[totSize*tNow+ipx] - phi[totSize*tNow+i]*phi[totSize*(nts+tNow)+ipx]) +
					sin(theta[totSize*tNow+i])*(phi[totSize*tNow+i]*phi[totSize*tNow+ipx] + phi[totSize*(nts+tNow)+i]*phi[totSize*(nts+tNow)+ipx]);

				cury = cos(theta[totSize*(nts+tNow)+i])*(phi[totSize*(nts+tNow)+i]*phi[totSize*tNow+ipy] - phi[totSize*tNow+i]*phi[totSize*(nts+tNow)+ipy]) +
					sin(theta[totSize*(nts+tNow)+i])*(phi[totSize*tNow+i]*phi[totSize*tNow+ipy] + phi[totSize*(nts+tNow)+i]*phi[totSize*(nts+tNow)+ipy]);

				curz = cos(theta[totSize*(nts*2+tNow)+i])*(phi[totSize*(nts+tNow)+i]*phi[totSize*tNow+ipz] - phi[totSize*tNow+i]*phi[totSize*(nts+tNow)+ipz]) +
					sin(theta[totSize*(nts*2+tNow)+i])*(phi[totSize*tNow+i]*phi[totSize*tNow+ipz] + phi[totSize*(nts+tNow)+i]*phi[totSize*(nts+tNow)+ipz]);


				// Calculate the derivatives of the field tensor (lattice version).

				Fxy_y = ( sin(theta[totSize*tNow+i] + theta[totSize*(nts+tNow)+ipx] - theta[totSize*tNow+ipy] - theta[totSize*(nts+tNow)+i]) - 
						sin(theta[totSize*tNow+imy] + theta[totSize*(nts+tNow)+ipxmy] - theta[totSize*tNow+i] - theta[totSize*(nts+tNow)+imy]) )/(dy*dy);

				Fxz_z = ( sin(theta[totSize*tNow+i] + theta[totSize*(nts*2+tNow)+ipx] - theta[totSize*tNow+ipz] - theta[totSize*(nts*2+tNow)+i]) -
						sin(theta[totSize*tNow+imz] + theta[totSize*(nts*2+tNow)+ipxmz] - theta[totSize*tNow+i] - theta[totSize*(nts*2+tNow)+imz]) )/(dz*dz);

				Fyx_x = ( sin(theta[totSize*(nts+tNow)+i] + theta[totSize*tNow+ipy] - theta[totSize*(nts+tNow)+ipx] - theta[totSize*tNow+i]) -
						sin(theta[totSize*(nts+tNow)+imx] + theta[totSize*tNow+imxpy] - theta[totSize*(nts+tNow)+i] - theta[totSize*tNow+imx]) )/(dx*dx);

				Fyz_z = ( sin(theta[totSize*(nts+tNow)+i] + theta[totSize*(nts*2+tNow)+ipy] - theta[totSize*(nts+tNow)+ipz] - theta[totSize*(nts*2+tNow)+i]) -
						sin(theta[totSize*(nts+tNow)+imz] + theta[totSize*(nts*2+tNow)+ipymz] - theta[totSize*(nts+tNow)+i] - theta[totSize*(nts*2+tNow)+imz]) )/(dz*dz);

				Fzx_x = ( sin(theta[totSize*(nts*2+tNow)+i] + theta[totSize*tNow+ipz] - theta[totSize*(nts*2+tNow)+ipx] - theta[totSize*tNow+i]) -
						sin(theta[totSize*(nts*2+tNow)+imx] + theta[totSize*tNow+imxpz] - theta[totSize*(nts*2+tNow)+i] - theta[totSize*tNow+imx]) )/(dx*dx);

				Fzy_y = ( sin(theta[totSize*(nts*2+tNow)+i] + theta[totSize*(nts+tNow)+ipz] - theta[totSize*(nts*2+tNow)+ipy] - theta[totSize*(nts+tNow)+i]) -
						sin(theta[totSize*(nts*2+tNow)+imy] + theta[totSize*(nts+tNow)+imypz] - theta[totSize*(nts*2+tNow)+i] - theta[totSize*(nts+tNow)+imy]) )/(dy*dy);

				// Calculate first order time derivatives
				thetat[0] = ( theta[totSize*tNow+i] - theta[totSize*tPast+i] )/dt;
				thetat[1] = ( theta[totSize*(nts+tNow)+i] - theta[totSize*(nts+tPast)+i] )/dt;
				thetat[2] = ( theta[totSize*(nts*2+tNow)+i] - theta[totSize*(nts*2+tPast)+i] )/dt;

				// Calculate second order time derivatives and update
				thetatt[0] = -pow( pow(tau,scaling), beta_g)*2*g*g*curx - Fxy_y - Fxz_z - fric*thetat[0];
				thetatt[1] = -pow( pow(tau,scaling), beta_g)*2*g*g*cury - Fyx_x - Fyz_z - fric*thetat[1];
				thetatt[2] = -pow( pow(tau,scaling), beta_g)*2*g*g*curz - Fzx_x - Fzy_y - fric*thetat[2];

				theta[totSize*tPast+i] = 2*theta[totSize*tNow+i] - theta[totSize*tPast+i] + dt*dt*thetatt[0];
				theta[totSize*(nts+tPast)+i] = 2*theta[totSize*(nts+tNow)+i] - theta[totSize*(nts+tPast)+i] + dt*dt*thetatt[1];
				theta[totSize*(nts*2+tPast)+i] = 2*theta[totSize*(nts*2+tNow)+i] - theta[totSize*(nts*2+tPast)+i] + dt*dt*thetatt[2];
			}

            if(calcEnergy){

                for(comp=0;comp<2;comp++){

                    phix[comp] = ( phi[totSize*(nts*comp+tNow)+i] - cos(theta[totSize*tNow+imx])*phi[totSize*(nts*comp+tNow)+imx] + c[comp]*sin(theta[totSize*tNow+imx])*phi[totSize*(nts*(comp+c[comp])+tNow)+imx] )/dx;
                    phiy[comp] = ( phi[totSize*(nts*comp+tNow)+i] - cos(theta[totSize*(nts+tNow)+imy])*phi[totSize*(nts*comp+tNow)+imy] + c[comp]*sin(theta[totSize*(nts+tNow)+imy])*phi[totSize*(nts*(comp+c[comp])+tNow)+imy] )/dy;
                    phiz[comp] = ( phi[totSize*(nts*comp+tNow)+i] - cos(theta[totSize*(nts*2+tNow)+imz])*phi[totSize*(nts*comp+tNow)+imz] + c[comp]*sin(theta[totSize*(nts*2+tNow)+imz])*phi[totSize*(nts*(comp+c[comp])+tNow)+imz] )/dz;

                }

                thetaDotCont = 0.5*( pow(thetat[0],2) + pow(thetat[1],2) + pow(thetat[2],2) ); 

                // Field strength terms calculated from Wilson loop (factor of half ignored as Fxy=Fyx)

                Fxy = ( 1 - cos(theta[totSize*tNow+i] + theta[totSize*(nts+tNow)+ipx] - theta[totSize*tNow+ipy] - theta[totSize*(nts+tNow)+i]) )/pow(g*dx*dy,2);
                Fxz = ( 1 - cos(theta[totSize*tNow+i] + theta[totSize*(nts*2+tNow)+ipx] - theta[totSize*tNow+ipz] - theta[totSize*(nts*2+tNow)+i]) )/pow(g*dx*dz,2);
                Fyz = ( 1 - cos(theta[totSize*(nts+tNow)+i] + theta[totSize*(nts*2+tNow)+ipy] - theta[totSize*(nts+tNow)+ipz] - theta[totSize*(nts*2+tNow)+i]) )/pow(g*dy*dz,2);

                FCont = Fxy + Fxz + Fyz;

                localEnergy += ( pow(phit[0],2) + pow(phit[1],2) + pow(phix[0],2) + pow(phix[1],2) + pow(phiy[0],2) + pow(phiy[1],2) + pow(phiz[0],2) + pow(phiz[1],2)
                                + 0.25*lambda*pow(phiMagSqr-pow(eta,2),2) )*dx*dy*dz;//+ thetaDotCont + FCont

				if((calc_energy_spectr_direct and (TimeStep>=damped_nt) and ((frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))or(TimeStep%saveenergyspecfreq==0 ))) or (frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))){
					
					double energy_v = 0.0;
					long int index = i-frontHaloSize;

					energy_v = (pow(phit[0], 2) + pow(phit[1], 2)); // change this if you need other forms of energy, currently set to kinetic energy
					local_Energy[index] = energy_v; //+ thetaDotCont + FCont

				}
                
				if((calc_energy_spectr_indirect and (TimeStep>=damped_nt) and ((frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))or(TimeStep%saveenergyspecfreq1==0 ))) or (frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))){
					double chi = sqrt(phiMagSqr) - eta;
					double chi_past = sqrt(phiMagSqr_past) - eta;
					double spectr_phase = std::atan2(phi[totSize*(nts+tNow)+i], phi[totSize*tNow+i]);
					double spectr_phase_past = std::atan2(phi[totSize*(nts+tPast)+i], phi[totSize*tPast+i]);
					long int index = i-frontHaloSize;
					
					// Check if real part is 0
					if(phi[totSize * tPast + i] == 0) {spectr_phase_past = M_PI / 2; } // Set spectr_phase_past to pi/2
					if(phi[totSize * tNow + i] == 0) {spectr_phase = M_PI / 2; }// Set spectr_phase to pi/2

					chi_array[index] = chi;
					chi_array_past[index] = chi_past;
					spectr_phase_array[index] = spectr_phase;
					spectr_phase_array_past[index] = spectr_phase_past;

				}

                if((calc_energy_density and (TimeStep>=damped_nt) and ((frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))or(TimeStep%energyDensityFreq==0 ))) or (frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))){
                    
                    double x,y,z;

                    x = ( (i+dataStart)/(ny*nz) - x0 )*dx;
                    y = ( ((i+dataStart)/nz)%ny - y0 )*dy;
                    z = ( (i+dataStart)%nz - z0 )*dz;

                    dens_en0.push_back(x);
                    dens_en1.push_back(y);
                    dens_en2.push_back(z);

                    if(output_energy_all){
                    dens_en3.push_back((pow(phit[0],2) + pow(phit[1],2))*dx*dy*dz); // kinetic energy
                    dens_en4.push_back((pow(phix[0],2) + pow(phix[1],2) + pow(phiy[0],2) + pow(phiy[1],2) + pow(phiz[0],2) + pow(phiz[1],2))*dx*dy*dz); // gradient energy
                    dens_en5.push_back((0.25*lambda*pow(phiMagSqr-pow(eta,2),2))*dx*dy*dz); // Potential energy
                    dens_en6.push_back((thetaDotCont)*dx*dy*dz); //for gauge fields
                    dens_en7.push_back((FCont)*dx*dy*dz);
                    dens_en8.push_back(phiMagSqr);
					}

					if(output_energy_total_potential){
					dens_en5.push_back((0.25*lambda*pow(phiMagSqr-pow(eta,2),2))*dx*dy*dz); // Potential energy
					dens_en8.push_back(phiMagSqr);
					dens_en9.push_back(pow(phit[0],2) + pow(phit[1],2) + pow(phix[0],2) + pow(phix[1],2) + pow(phiy[0],2) + pow(phiy[1],2) + pow(phiz[0],2) + pow(phiz[1],2)
                                + 0.25*lambda*pow(phiMagSqr-pow(eta,2),2));
					}
                }
            }	

			if(((calc_spectr and (TimeStep>=damped_nt) and ((frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))or(TimeStep%savespecfreq==0 ))) or (frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1)))or(ro_a and (TimeStep>=damped_nt) and (TimeStep%savespecfreq==0 ))){
				double axion_v = 0.0;
				long int index = i-frontHaloSize;
				axion_v = (phi[totSize*(tNow)+i]*phit[1] - phi[totSize*(nts+tNow)+i]*phit[0])/(pow(phi[totSize*(tNow)+i],2)+pow(phi[totSize*(nts+tNow)+i],2));
				ro_a_sum = ro_a_sum + pow(axion_v,2)*dx*dy*dz;
				axion[index] = axion_v;
				phi1_out[index] = phi[totSize*(tNow)+i];
				phi2_out[index] = phi[totSize*(nts+tNow)+i];
			}

	        if(stringDetect and (!detectBuffer or TimeStep>=damped_nt) and ((frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))or(TimeStep%saveFreq==0 ))){

	        	double x,y,z,coeff1[2],coeff2[2],coeff3[2],coeff4[2],a,b,c,discrim,sol1,sol2;
	        	long long int ipxpy,ipxpz,ipypz;

	        	ipxpy = ipy+ny*nz;
	        	ipxpz = ipz+ny*nz;
	        	ipypz = (ipy+dataStart+1)%nz + ( (ipy+dataStart)/nz )*nz - dataStart;

	        	x = ( (i+dataStart)/(ny*nz) - x0 )*dx;
	        	y = ( ((i+dataStart)/nz)%ny - y0 )*dy;
	        	z = ( (i+dataStart)%nz - z0 )*dz;

		        for(comp=0;comp<2;comp++){

		            coeff1[comp] = phi[totSize*(nts*comp+tNow)+ipxpy] - phi[totSize*(nts*comp+tNow)+ipx] - phi[totSize*(nts*comp+tNow)+ipy] + phi[totSize*(nts*comp+tNow)+i];
		            coeff2[comp] = (y+dy)*(phi[totSize*(nts*comp+tNow)+ipx] - phi[totSize*(nts*comp+tNow)+i]) + y*(phi[totSize*(nts*comp+tNow)+ipy] - phi[totSize*(nts*comp+tNow)+ipxpy]);
		            coeff3[comp] = (x+dx)*(phi[totSize*(nts*comp+tNow)+ipy] - phi[totSize*(nts*comp+tNow)+i]) + x*(phi[totSize*(nts*comp+tNow)+ipx] - phi[totSize*(nts*comp+tNow)+ipxpy]);
		            coeff4[comp] = (x+dx)*( (y+dy)*phi[totSize*(nts*comp+tNow)+i] - y*phi[totSize*(nts*comp+tNow)+ipy] ) - x*( (y+dy)*phi[totSize*(nts*comp+tNow)+ipx] - y*phi[totSize*(nts*comp+tNow)+ipxpy] );

		        }

		        // Substituting one equation into the other gives a quadratic equation for one of the coordinates. Now calculate the coefficients of the quadratic.

		        a = coeff1[1]*coeff3[0] - coeff1[0]*coeff3[1];  
		        b = coeff1[1]*coeff4[0] + coeff2[1]*coeff3[0] - coeff1[0]*coeff4[1] - coeff2[0]*coeff3[1];
		        c = coeff2[1]*coeff4[0] - coeff2[0]*coeff4[1];

		        discrim = b*b - 4*a*c;

		        // Check if a=0, if so the equation is simpler than quadratic --> sol2 (y in this case) is just -c/b unless b is zero as well
		        if(a==0){
		            if(b!=0){

		                // There is just one solution

		                sol2 = -c/b; // y location of string

		                // Does this lie inside the face?
		                if(sol2>=y && sol2<=y+dy){

		                    sol1 = -(coeff3[0]*sol2 + coeff4[0])/(coeff1[0]*sol2 + coeff2[0]); // x location of string

		                    // Does this lie inside the face?
		                    if(sol1>=x && sol1<=x+dx){

		                        // The string intersects the face so add it to the vectors

		                        xString.push_back(sol1);
		                        yString.push_back(sol2);
		                        zString.push_back(z);

		                    }

		                }

		            }

		            // Otherwise no solutions

		        } else if(discrim >= 0){

		            // There will be two solutions (or a repeated, this may cause trouble). First solution is

		            sol2 = ( -b + sqrt(discrim) )/(2*a);

		            if(sol2>=y && sol2<=y+dy){

		                sol1 = -(coeff3[0]*sol2 + coeff4[0])/(coeff1[0]*sol2 + coeff2[0]);

		                if(sol1>=x && sol1<=x+dx){

		                    xString.push_back(sol1);
		                    yString.push_back(sol2);
		                    zString.push_back(z);

		                }

		            }

		            // Second solution is

		            sol2 = ( -b - sqrt(discrim) )/(2*a);

		            if(sol2>=y && sol2<=y+dy){

		                sol1 = -(coeff3[0]*sol2 + coeff4[0])/(coeff1[0]*sol2 + coeff2[0]);

		                if(sol1>=x && sol1<=x+dx){

		                    xString.push_back(sol1);
		                    yString.push_back(sol2);
		                    zString.push_back(z);

		                }

		            }

		        }

		        // Now repeat this process for the y directed face

	            for(comp=0;comp<2;comp++){

	                coeff1[comp] = phi[totSize*(nts*comp+tNow)+ipxpz] - phi[totSize*(nts*comp+tNow)+ipx] - phi[totSize*(nts*comp+tNow)+ipz] + phi[totSize*(nts*comp+tNow)+i];
	                coeff2[comp] = (z+dz)*(phi[totSize*(nts*comp+tNow)+ipx] - phi[totSize*(nts*comp+tNow)+i]) + z*(phi[totSize*(nts*comp+tNow)+ipz] - phi[totSize*(nts*comp+tNow)+ipxpz]);
	                coeff3[comp] = (x+dx)*(phi[totSize*(nts*comp+tNow)+ipz] - phi[totSize*(nts*comp+tNow)+i]) + x*(phi[totSize*(nts*comp+tNow)+ipx] - phi[totSize*(nts*comp+tNow)+ipxpz]);
	                coeff4[comp] = (x+dx)*( (z+dz)*phi[totSize*(nts*comp+tNow)+i] - z*phi[totSize*(nts*comp+tNow)+ipz] ) - x*( (z+dz)*phi[totSize*(nts*comp+tNow)+ipx] - z*phi[totSize*(nts*comp+tNow)+ipxpz] );

	            }

	            a = coeff1[1]*coeff3[0] - coeff1[0]*coeff3[1];
	            b = coeff1[1]*coeff4[0] + coeff2[1]*coeff3[0] - coeff1[0]*coeff4[1] - coeff2[0]*coeff3[1];
	            c = coeff2[1]*coeff4[0] - coeff2[0]*coeff4[1];

	            discrim = b*b - 4*a*c;

	            if(a==0){
	                if(b!=0){

	                    sol2 = -c/b;

	                    if(sol2>=z && sol2<=z+dz){

	                        sol1 = -(coeff3[0]*sol2 + coeff4[0])/(coeff1[0]*sol2 + coeff2[0]);

	                        if(sol1>=x && sol1<=x+dx){

	                            xString.push_back(sol1);
	                            yString.push_back(y);
	                            zString.push_back(sol2);

	                        }

	                    }

	                }
	            } else if(discrim >= 0){

	                sol2 = ( -b + sqrt(discrim) )/(2*a);

	                if(sol2>=z && sol2<=z+dz){

	                    sol1 = -(coeff3[0]*sol2 + coeff4[0])/(coeff1[0]*sol2 + coeff2[0]);

	                    if(sol1>=x && sol1<=x+dx){

	                        xString.push_back(sol1);
	                        yString.push_back(y);
	                        zString.push_back(sol2);

	                    }

	                }

	                sol2 = ( -b - sqrt(discrim) )/(2*a);

	                if(sol2>=z && sol2<=z+dz){

	                    sol1 = -(coeff3[0]*sol2 + coeff4[0])/(coeff1[0]*sol2 + coeff2[0]);

	                    if(sol1>=x && sol1<=x+dx){

	                        xString.push_back(sol1);
	                        yString.push_back(y);
	                        zString.push_back(sol2);

	                    }

	                }

	            }



	            // Now repeat one more time for the x directed face

	            for(comp=0;comp<2;comp++){

	                coeff1[comp] = phi[totSize*(nts*comp+tNow)+ipypz] - phi[totSize*(nts*comp+tNow)+ipy] - phi[totSize*(nts*comp+tNow)+ipz] + phi[totSize*(nts*comp+tNow)+i];
	                coeff2[comp] = (z+dz)*(phi[totSize*(nts*comp+tNow)+ipy] - phi[totSize*(nts*comp+tNow)+i]) + z*(phi[totSize*(nts*comp+tNow)+ipz] - phi[totSize*(nts*comp+tNow)+ipypz]);
	                coeff3[comp] = (y+dy)*(phi[totSize*(nts*comp+tNow)+ipz] - phi[totSize*(nts*comp+tNow)+i]) + y*(phi[totSize*(nts*comp+tNow)+ipy] - phi[totSize*(nts*comp+tNow)+ipypz]);
	                coeff4[comp] = (y+dy)*( (z+dz)*phi[totSize*(nts*comp+tNow)+i] - z*phi[totSize*(nts*comp+tNow)+ipz] ) - y*( (z+dz)*phi[totSize*(nts*comp+tNow)+ipy] - z*phi[totSize*(nts*comp+tNow)+ipypz] );

	            }

	            a = coeff1[1]*coeff3[0] - coeff1[0]*coeff3[1];
	            b = coeff1[1]*coeff4[0] + coeff2[1]*coeff3[0] - coeff1[0]*coeff4[1] - coeff2[0]*coeff3[1];
	            c = coeff2[1]*coeff4[0] - coeff2[0]*coeff4[1];

	            discrim = b*b - 4*a*c;

	            if(a==0){
	                if(b!=0){

	                    sol2 = -c/b;

	                    if(sol2>=z && sol2<=z+dz){

	                        sol1 = -(coeff3[0]*sol2 + coeff4[0])/(coeff1[0]*sol2 + coeff2[0]);

	                        if(sol1>=y && sol1<=y+dy){

	                            xString.push_back(x);
	                            yString.push_back(sol1);
	                            zString.push_back(sol2);

	                        }

	                    }

	                }
	            } else if(discrim >= 0){

	                sol2 = ( -b + sqrt(discrim) )/(2*a);

	                if(sol2>=z && sol2<=z+dz){

	                    sol1 = -(coeff3[0]*sol2 + coeff4[0])/(coeff1[0]*sol2 + coeff2[0]);

	                    if(sol1>=y && sol1<=y+dy){

	                        xString.push_back(x);
	                        yString.push_back(sol1);
	                        zString.push_back(sol2);

	                    }

	                }

	                sol2 = ( -b - sqrt(discrim) )/(2*a);

	                if(sol2>=z && sol2<=z+dz){

	                    sol1 = -(coeff3[0]*sol2 + coeff4[0])/(coeff1[0]*sol2 + coeff2[0]);

	                    if(sol1>=y && sol1<=y+dy){

	                        xString.push_back(x);
	                        yString.push_back(sol1);
	                        zString.push_back(sol2);

	                    }

	                }

	            }

		    }

        }
		
		if((ro_a and (TimeStep%savespecfreq==0 ))){
			if(rank == 0){
				long double sum_ro_a = 0.0;
				sum_ro_a = sum_ro_a + ro_a_sum;


	        	string ro_Path = dir_path + "/GifData/ro_Data_" + outTag + "_nx" + to_string(nx) + "_dampednt" + to_string(damped_nt) + "_ntHeld" + to_string(ntHeld) + "_seed" + to_string(seed) + "_" + to_string(TimeStep) + ".txt";
	            ofstream ro_Data (ro_Path.c_str());
				
				for(i=1;i<size;i++){

					long double ro_a_sum_rec;
	        	
	        		MPI_Recv(&ro_a_sum_rec,1,MPI_DOUBLE,i,1000,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	        		sum_ro_a = sum_ro_a + ro_a_sum_rec;
	        	}

				ro_Data << (sum_ro_a/(dx*nx*dy*ny*dz*nz)) << endl;

			}else{

	        	MPI_Send(&ro_a_sum,1,MPI_INT,0,1000,MPI_COMM_WORLD);

	        }


		}

        // Next step is to collect all the intersections together on the master process and output them to text

        if(stringDetect and (!detectBuffer or TimeStep>=damped_nt) and ((frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))or(TimeStep%saveFreq==0 ))){

	        if(rank==0){

	        	// Output the string crossings of each process one at a time to prevent the memory requirements on 1 node from getting too large.

	        	int numTotalIntersections = xString.size();

	        	ss.str(string());
	        	ss << gifStringPosFrame;
	        	string gifStringPosDataPath = dir_path + "/GifData/gifStringPosData_" + outTag + "_nx" + to_string(nx) + "_dampednt" + to_string(damped_nt) + "_ntHeld" + to_string(ntHeld) + "_seed" + to_string(seed) + "_" + to_string(TimeStep) + ".txt";
	            ofstream gifStringPosData (gifStringPosDataPath.c_str());
	            gifStringPosFrame+=1;

	            // Output the data from the master process

	            for(i=0;i<xString.size();i++){

	            	gifStringPosData << xString[i] << " " << yString[i] << " " << zString[i] << endl;

	            }

	        	for(i=1;i<size;i++){

	        		// Find out how many intersections each process found

	        		int nIntersections;

	        		MPI_Recv(&nIntersections,1,MPI_INT,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	        		numTotalIntersections += nIntersections;

	        		// Declare vectors to receive each process's intersections and receive the data.

	        		vector<double> xString(nIntersections,0.0), yString(nIntersections,0.0), zString(nIntersections,0.0);

	        		MPI_Recv(&xString[0],nIntersections,MPI_DOUBLE,i,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	        		MPI_Recv(&yString[0],nIntersections,MPI_DOUBLE,i,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	        		MPI_Recv(&zString[0],nIntersections,MPI_DOUBLE,i,4,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	        		for(j=0;j<xString.size();j++){

	        			gifStringPosData << xString[j] << " " << yString[j] << " " << zString[j] << endl;

	        		}

	        	}

	        	if(numTotalIntersections==0){ stringsExist = false; }

	        } else{

	        	// Tell the master process how many string intersections to expect
	        	int nIntersections = xString.size();
	        	MPI_Send(&nIntersections,1,MPI_INT,0,1,MPI_COMM_WORLD);

	        	// Send the data
	        	MPI_Send(&xString[0],nIntersections,MPI_DOUBLE,0,2,MPI_COMM_WORLD);
	        	MPI_Send(&yString[0],nIntersections,MPI_DOUBLE,0,3,MPI_COMM_WORLD);
	        	MPI_Send(&zString[0],nIntersections,MPI_DOUBLE,0,4,MPI_COMM_WORLD);

	        }

	    }

        if(calc_energy_density and (!detectBuffer or TimeStep>=damped_nt) and ((frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1))or(TimeStep%energyDensityFreq==0 ))){
            
            if(rank==0){

                int numener = dens_en1.size();

                string energygifdata = dir_path + "/DS/Energydensity_"  + outTag + "_nx" + to_string(nx) + "_dampednt" + to_string(damped_nt) + "_ntHeld" + to_string(ntHeld) + "_seed" + to_string(seed) + "_" + to_string(TimeStep) + ".txt";
                ofstream gifenergyPosData (energygifdata.c_str());

                if(output_energy_all){
                for(i=0;i<dens_en0.size();i++){
                    gifenergyPosData << dens_en0[i] << " " << dens_en1[i] << " " << dens_en2[i] <<" " << dens_en3[i] << " " << dens_en4[i] << " " << dens_en5[i] <<" " << dens_en6[i] << " " << dens_en7[i] << " " << dens_en8[i] <<endl;
                	}
				}
				else if(output_energy_total_potential){
					for(i=0;i<dens_en0.size();i++){
                    gifenergyPosData << dens_en0[i] << " " << dens_en1[i] << " " << dens_en2[i] <<" " << dens_en5[i] << " " << dens_en9[i] << " " << dens_en8[i] <<endl;
            		}
				}

                for(i=1;i<size;i++){

                    int nener;
                    MPI_Recv(&nener,1,MPI_INT,i,5,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    numener += nener;
                    vector<double> dens_en0(nener,0.0), dens_en1(nener,0.0), dens_en2(nener,0.0), dens_en3(nener,0.0), dens_en4(nener,0.0), dens_en5(nener,0.0), dens_en6(nener,0.0), dens_en7(nener,0.0), dens_en8(nener,0.0), dens_en9(nener,0.0);

                    if(output_energy_all){
						MPI_Recv(&dens_en0[0],nener,MPI_DOUBLE,i,6,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en1[0],nener,MPI_DOUBLE,i,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en2[0],nener,MPI_DOUBLE,i,8,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en3[0],nener,MPI_DOUBLE,i,9,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en4[0],nener,MPI_DOUBLE,i,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en5[0],nener,MPI_DOUBLE,i,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en6[0],nener,MPI_DOUBLE,i,12,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en7[0],nener,MPI_DOUBLE,i,13,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en8[0],nener,MPI_DOUBLE,i,17,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

						for(p=0;p<dens_en0.size();p++){
                        gifenergyPosData << dens_en0[p] << " " << dens_en1[p] << " " << dens_en2[p] <<" " << dens_en3[p] << " " << dens_en4[p] << " " << dens_en5[p] <<" " << dens_en6[p] << " " << dens_en7[p] << " " << dens_en8[p] <<endl;
	                    }
					}

					else if(output_energy_total_potential){
						MPI_Recv(&dens_en0[0],nener,MPI_DOUBLE,i,6,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en1[0],nener,MPI_DOUBLE,i,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en2[0],nener,MPI_DOUBLE,i,8,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en5[0],nener,MPI_DOUBLE,i,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en9[0],nener,MPI_DOUBLE,i,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&dens_en8[0],nener,MPI_DOUBLE,i,17,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						for(p=0;p<dens_en0.size();p++){
                        gifenergyPosData << dens_en0[p] << " " << dens_en1[p] << " " << dens_en2[p] << " " << dens_en5[p] <<" " << dens_en9[p]  <<" " << dens_en8[p] <<endl;
	                    }
					}

                }
            } else{

                int nener = dens_en0.size();
                MPI_Send(&nener,1,MPI_INT,0,5,MPI_COMM_WORLD);

                if(output_energy_all){
					MPI_Send(&dens_en0[0],nener,MPI_DOUBLE,0,6,MPI_COMM_WORLD);
					MPI_Send(&dens_en1[0],nener,MPI_DOUBLE,0,7,MPI_COMM_WORLD);
					MPI_Send(&dens_en2[0],nener,MPI_DOUBLE,0,8,MPI_COMM_WORLD);
					MPI_Send(&dens_en3[0],nener,MPI_DOUBLE,0,9,MPI_COMM_WORLD);
					MPI_Send(&dens_en4[0],nener,MPI_DOUBLE,0,10,MPI_COMM_WORLD);
					MPI_Send(&dens_en5[0],nener,MPI_DOUBLE,0,11,MPI_COMM_WORLD);
					MPI_Send(&dens_en6[0],nener,MPI_DOUBLE,0,12,MPI_COMM_WORLD);
					MPI_Send(&dens_en7[0],nener,MPI_DOUBLE,0,13,MPI_COMM_WORLD);
					MPI_Send(&dens_en8[0],nener,MPI_DOUBLE,0,17,MPI_COMM_WORLD);
				}

				else if(output_energy_total_potential){
					MPI_Send(&dens_en0[0],nener,MPI_DOUBLE,0,6,MPI_COMM_WORLD);
					MPI_Send(&dens_en1[0],nener,MPI_DOUBLE,0,7,MPI_COMM_WORLD);
					MPI_Send(&dens_en2[0],nener,MPI_DOUBLE,0,8,MPI_COMM_WORLD);
					MPI_Send(&dens_en5[0],nener,MPI_DOUBLE,0,11,MPI_COMM_WORLD);
					MPI_Send(&dens_en9[0],nener,MPI_DOUBLE,0,11,MPI_COMM_WORLD);
					MPI_Send(&dens_en8[0],nener,MPI_DOUBLE,0,17,MPI_COMM_WORLD);
           		}

            }
            
        }
		//------------------------------------------------------------------------------------------------------------------------------------------------------

		//------------------------------------------------------------------------------------------------------------------------------------------------------

		//------------------------------------------------------------------------------------------------------------------------------------------------------

		if(frame_out and (std::count(frames.begin(), frames.end(), TimeStep) == 1)){

			if(output_mode == 1){
				if(rank==0){
					int numTotaltheta = coreSize;
					
					string thetata_dir = dir_path + "/field_axion_nx" + to_string(nx) + "_seed" + to_string(seed) + "the_iteration" + to_string(TimeStep) + ".txt";
					ofstream thetataData (thetata_dir.c_str());
					
					
					for(i=0;i<coreSize;i++){
						double x,y,z;
						x = ( (i)/(ny*nz) - x0 )*dx;
						y = ( ((i)/nz)%ny - y0 )*dy;
						z = ( (i)%nz - z0 )*dz;
						thetataData << x << "  "<< y <<"  "<< z <<"  "<< axion[i] <<"  "<<phi1_out[i]<<" " <<phi2_out[i]<< endl;
					
					}

					for(i=1;i<size;i++){
						int nInter;
						int start_c;
						MPI_Recv(&nInter,1,MPI_INT,i,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&start_c,1,MPI_INT,i,12,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						numTotaltheta += nInter;
						
						vector<double> axionn(nInter, 0.0),phi1_outt(nInter, 0.0),phi2_outt(nInter, 0.0);
						
						MPI_Recv(&axionn[0],nInter,MPI_DOUBLE,i,15,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&phi1_outt[0],nInter,MPI_DOUBLE,i,16,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&phi2_outt[0],nInter,MPI_DOUBLE,i,17,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

						for(j=0;j<coreSize;j++){
							double x,y,z;
							x = ( (j + start_c)/(ny*nz) - x0 )*dx;
							y = ( ((j + start_c)/nz)%ny - y0 )*dy;
							z = ( (j + start_c)%nz - z0 )*dz;
							thetataData << x << " " << y << " " << z << " " << axionn[j]<<"  " <<phi1_outt[j]<<" " <<phi2_outt[j]<< endl;
						}
					}
				} else{       	
					int nInter = coreSize;
					int start_c = coreStart;
					MPI_Send(&nInter,1,MPI_INT,0,11,MPI_COMM_WORLD);
					MPI_Send(&start_c,1,MPI_INT,0,12,MPI_COMM_WORLD);

					MPI_Send(&axion[0],nInter,MPI_DOUBLE,0,15,MPI_COMM_WORLD);
					MPI_Send(&phi1_out[0],nInter,MPI_DOUBLE,0,16,MPI_COMM_WORLD);
					MPI_Send(&phi2_out[0],nInter,MPI_DOUBLE,0,17,MPI_COMM_WORLD);
				}

			}

			//------------------------------------------------------------------------------------------------------------------------------------------------------

			if(output_mode == 0){
				if(rank==0){
					int numTotaltheta = coreSize;
					
					string thetata_dir = dir_path + "/field_axion_nx" + to_string(nx) + "_seed" + to_string(seed) + "the_iteration" + to_string(TimeStep) + ".txt";
					ofstream thetataData (thetata_dir.c_str());
					
					
					for(i=0;i<coreSize;i++){
						
						thetataData << axion[i] <<"  "<<phi1_out[i]<<" " <<phi2_out[i]<< endl;
					
					}

					for(i=1;i<size;i++){
						int nInter;
						
						MPI_Recv(&nInter,1,MPI_INT,i,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						
						numTotaltheta += nInter;
						
						vector<double> axionn(nInter, 0.0),phi1_outt(nInter, 0.0),phi2_outt(nInter, 0.0);
						
						MPI_Recv(&axionn[0],nInter,MPI_DOUBLE,i,15,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&phi1_outt[0],nInter,MPI_DOUBLE,i,16,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						MPI_Recv(&phi2_outt[0],nInter,MPI_DOUBLE,i,17,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

						for(j=0;j<coreSize;j++){

							thetataData << axionn[j]<<"  " <<phi1_outt[j]<<" " <<phi2_outt[j]<< endl;
						}
					}
				} else{       	
					int nInter = coreSize;
					
					MPI_Send(&nInter,1,MPI_INT,0,11,MPI_COMM_WORLD);
					

					MPI_Send(&axion[0],nInter,MPI_DOUBLE,0,15,MPI_COMM_WORLD);
					MPI_Send(&phi1_out[0],nInter,MPI_DOUBLE,0,16,MPI_COMM_WORLD);
					MPI_Send(&phi2_out[0],nInter,MPI_DOUBLE,0,17,MPI_COMM_WORLD);
				}

			}
		}

		//------------------------------------------------------------------------------------------------------------------------------------------------------

        if(calcEnergy){

            if(rank==0){

                double energy = localEnergy; 

                for(i=1;i<size;i++){ MPI_Recv(&localEnergy,1,MPI_DOUBLE,i,5,MPI_COMM_WORLD,MPI_STATUS_IGNORE);  energy += localEnergy; }

                valsPerLoop << energy << endl;

            } else{ MPI_Send(&localEnergy,1,MPI_DOUBLE,0,5,MPI_COMM_WORLD); }

        }

        for(comp=0;comp<2;comp++){ 

        	MPI_Sendrecv(&phi[totSize*(nts*comp+tPast)+frontHaloSize],nbrBackHaloSize,MPI_DOUBLE,(rank-1+size)%size,0, 
        				 &phi[totSize*(nts*comp+tPast)+coreSize+frontHaloSize],backHaloSize,MPI_DOUBLE,(rank+1)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 

        	MPI_Sendrecv(&phi[totSize*(nts*comp+tPast)+coreSize+frontHaloSize-nbrFrontHaloSize],nbrFrontHaloSize,MPI_DOUBLE,(rank+1)%size,0,
        				 &phi[totSize*(nts*comp+tPast)],frontHaloSize,MPI_DOUBLE,(rank-1+size)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        }

        for(comp=0;comp<3;comp++){ 

        	MPI_Sendrecv(&theta[totSize*(nts*comp+tPast)+frontHaloSize],nbrBackHaloSize,MPI_DOUBLE,(rank-1+size)%size,0,
        				 &theta[totSize*(nts*comp+tPast)+coreSize+frontHaloSize],backHaloSize,MPI_DOUBLE,(rank+1)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        	MPI_Sendrecv(&theta[totSize*(nts*comp+tPast)+coreSize+frontHaloSize-nbrFrontHaloSize],nbrFrontHaloSize,MPI_DOUBLE,(rank+1)%size,0,
        				 &theta[totSize*(nts*comp+tPast)],frontHaloSize,MPI_DOUBLE,(rank-1+size)%size,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        }

    MPI_Barrier(MPI_COMM_WORLD);

    }

	//------------------------------------------------------------------------------------------------------------------------------------------------------

    if(rank==0){

	    cout << "\rTimestep " << nt << " completed." << endl;

	    gettimeofday(&end,NULL);

	    cout << "Time taken: " << end.tv_sec - start.tv_sec << "s" << endl;

	}


    MPI_Finalize();


	return 0;

}

//////////////////////////////////////////////////////////////////////// END //////////////////////////////////////////////////////////////////////////////////////////////////////////////////