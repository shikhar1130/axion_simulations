#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <complex>
#include <random>

#include "array.hpp"

using namespace std;
typedef complex<double> dcmplx;

// Function Declarations
Array ProcessLoop(double xb2, double yb2, double zb2, int i, int j, int k);
Array ProcessLooper(double xb1, double yb1, double zb1, int i, int j, int k);
Array ProcessLoop2(double xb3, double yb3, double zb3, int i, int j, int k);
Array ProcessLooper2(double xb4, double yb4, double zb4, int i, int j, int k);

// Perturb a straight, global string solution, evolve the system and investigate how it radiates.


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                 		  Parameters & Declarations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const string ic_type = "loop collision";	 // Which type of initial condition generation to use. "NG sine" bases initial conditions on the Nambu-Goto sine wave solution constructed with straight string solutions 
// "simple sine" offsets the straight string solutions (x position) by a sine wave
// "random" creates random initial conditions. "boost" creates a single straight (z directed) string with a Lorentz boost applied.
            // "loop collision" creates two sets of (seperated) string, anti-string pairs. They are boosted towards each other and patched together so that they will collide
            // and form a loop (2 one due to periodic boundary conditions which are required for this sim) 

const int nx = 101;
const int ny = 101;
const int nz = 101;
const double dx = 0.25;
const double dy = 0.25;
const double dz = 0.25;
const double dt = 0.05;

const int n = 1; // This is useless for now, code assumes it is 1.
const double g = 0;

const double pi = 4*atan(1);

// Needed for straight string solutions based initial conditions /////////////////////

const int SORnx = 30001;
const double SORa = 0.01;

// Needed for perturbed NG straight strings (i.e NG sine) ///////////////////////////

const double tol = 1e-6;
const int N_ellipint = 1001; // Number of points to use for elliptical integral evaluation

//const double eps = 0.1;
//const double lambda = dz*(nz-1)/(1-0.25*eps*eps);

const double gr = 0.5*(1+sqrt(5));


// Needed for random initial conditions ///////////////////////////////////////////////

const double seed = 42;
const double mean = 0; // Probably always want this to be zero
const double stdev = 0.5;


// Needed for the case of one z directed boosted string

// Relativistic velocities (c=1) so vx^2+vy^2+vz^2 must be less than 1

const double v1x = 0.6*0.4;
const double v1y = 0.6*sqrt(1-pow(0.4,2));
const double v1z = 0;

// Needed for loop collision. Assumes string anti-string pair 1 are x directed strings and pair 2 are z directed strings.


const double pos1s[2] = {0.15*(ny-1)*dy, -0.25*(nz-1)*dz}; // Old {2, -0.25*(nz-1)*dz}; // y and z coordinates
const double pos1a[2] = {-0.15*(ny-1)*dy, 0.25*(nz-1)*dz}; // Old {-2, 0.25*(nz-1)*dz};

const double pos2s[2] = {0.25*(nx-1)*dx, 0}; // Old {0.25*(nx-1)*dx, -1}; // x and y coordinates
const double pos2a[2] = {-0.25*(nx-1)*dx, 0}; // Old {-0.25*(nx-1)*dx, 1}; 


const double v1s[3] = {0, -0.75, 0}; 
const double v1a[3] = {0, 0.75, 0}; // Old {0, 0.6*sqrt(1-pow(0.4,2)), 0.6*0.4};

const double v2s[3] = {0, 0, 0}; // Old {0.6*0.4, 0.6*sqrt(1-pow(0.4,2)), 0};
const double v2a[3] = {0, 0, 0}; // Old {-0.6*0.4, -0.6*sqrt(1-pow(0.4,2)), 0};

const double omega = 0.5; // Phase modification parameter. Phase goes to zero more quickly for larger values. Old method
const double Lmod = 25; // Phase modification parameter. Length scale associated with modification. Old method


Array SOR_Fields(SORnx, 2, 0.0);

double distance2[4], phiMag[4], AMag[4], xs2s, xs2a, ys1s, ys1a, ys2s, ys2a, zs1s, zs1a, A1s[3], A1a[3], A2s[3], A2a[3], phi1s[2], phi1a[2], phi2s[2], phi2a[2],
A1sPatch[3], A1aPatch[3], A2sPatch[3], A2aPatch[3], phase_fac, norm_fac, phi1Intrp, phi2Intrp, phi1_2[2], phi2_2[2], v1sMagSqr, v1aMagSqr, v2sMagSqr, v2aMagSqr,
gamma1s, gamma1a, gamma2s, gamma2a;
// xb, yb and zb defined later

int pClosest[4], str, ip, im, jp, jm, kp, km;
dcmplx ci(0.0, 1.0), cUnitPos1s, cUnitPos1a, cUnitPos2s, cUnitPos2a, gRot1s, gRot1a, gRot2s, gRot2a;

string dir_path = "/share/c2/RAB_mphys/agarwal/loops/101";
string SOR_inputPath = dir_path + "/Data/SOR_Fields.txt";

ifstream SOR_input(SOR_inputPath.c_str());

const int x0 = round((nx-1)/2);
const int y00 = round((ny-1)/2);
const int z0 = round((nz-1)/2);

double sf_pos1s[2], sf_pos1a[2], sf_pos2s[2], sf_pos2a[2];

Array ProcessLoop(double xb2, double yb2, double zb2, int i, int j, int k) {

    v1sMagSqr = pow(v1s[0],2) + pow(v1s[1],2) + pow(v1s[2],2);
    v1aMagSqr = pow(v1a[0],2) + pow(v1a[1],2) + pow(v1a[2],2);
    v2sMagSqr = pow(v2s[0],2) + pow(v2s[1],2) + pow(v2s[2],2);
    v2aMagSqr = pow(v2a[0],2) + pow(v2a[1],2) + pow(v2a[2],2);

    gamma1s = 1/sqrt(1 - v1sMagSqr);
    gamma1a = 1/sqrt(1 - v1aMagSqr);
    gamma2s = 1/sqrt(1 - v2sMagSqr);
    gamma2a = 1/sqrt(1 - v2aMagSqr);

    if(v1sMagSqr==0){ sf_pos1s[0] = pos1s[0];  sf_pos1s[1] = pos1s[1]; }
    else{ sf_pos1s[0] = pos1s[0] + (gamma1s-1)*(v1s[1]*pos1s[0] + v1s[2]*pos1s[1])*v1s[1]/v1sMagSqr;  sf_pos1s[1] = pos1s[1] + (gamma1s-1)*(v1s[1]*pos1s[0] + v1s[2]*pos1s[1])*v1s[2]/v1sMagSqr; }

    if(v1aMagSqr==0){ sf_pos1a[0] = pos1a[0];  sf_pos1a[1] = pos1a[1]; }
    else{ sf_pos1a[0] = pos1a[0] + (gamma1a-1)*(v1a[1]*pos1a[0] + v1a[2]*pos1a[1])*v1a[1]/v1aMagSqr;  sf_pos1a[1] = pos1a[1] + (gamma1a-1)*(v1a[1]*pos1a[0] + v1a[2]*pos1a[1])*v1a[2]/v1aMagSqr; }

    if(v2sMagSqr==0){ sf_pos2s[0] = pos2s[0];  sf_pos2s[1] = pos2s[1]; }
    else{ sf_pos2s[0] = pos2s[0] + (gamma2s-1)*(v2s[0]*pos2s[0] + v2s[1]*pos2s[1])*v2s[0]/v2sMagSqr;  sf_pos2s[1] = pos2s[1] + (gamma2s-1)*(v2s[0]*pos2s[0] + v2s[1]*pos2s[1])*v2s[1]/v2sMagSqr; }

    if(v2aMagSqr==0){ sf_pos2a[0] = pos2a[0];  sf_pos2a[1] = pos2a[1]; }
    else{ sf_pos2a[0] = pos2a[0] + (gamma2a-1)*(v2a[0]*pos2a[0] + v2a[1]*pos2a[1])*v2a[0]/v2aMagSqr;  sf_pos2a[1] = pos2a[1] + (gamma2a-1)*(v2a[0]*pos2a[0] + v2a[1]*pos2a[1])*v2a[1]/v2aMagSqr; }


    // Firstly need to calculate what these positions correspond to in the frame with a stationary string at t=0.

    if(v1sMagSqr==0){ ys1s = yb2;  zs1s = zb2; }
    else{ ys1s = yb2 + (gamma1s-1)*(v1s[0]*xb2 + v1s[1]*yb2 + v1s[2]*zb2)*v1s[1]/v1sMagSqr;  zs1s = zb2 + (gamma1s-1)*(v1s[0]*xb2 + v1s[1]*yb2 + v1s[2]*zb2)*v1s[2]/v1sMagSqr; }

    if(v1aMagSqr==0){ ys1a = yb2;  zs1a = zb2; }
    else{ ys1a = yb2 + (gamma1a-1)*(v1a[0]*xb2 + v1a[1]*yb2 + v1a[2]*zb2)*v1a[1]/v1aMagSqr;  zs1a = zb2 + (gamma1a-1)*(v1a[0]*xb2 + v1a[1]*yb2 + v1a[2]*zb2)*v1a[2]/v1aMagSqr; }

    if(v2sMagSqr==0){ xs2s = xb2;  ys2s = yb2; }
    else{ xs2s = xb2 + (gamma2s-1)*(v2s[0]*xb2 + v2s[1]*yb2 + v2s[2]*zb2)*v2s[0]/v2sMagSqr;  ys2s = yb2 + (gamma2s-1)*(v2s[0]*xb2 + v2s[1]*yb2 + v2s[2]*zb2)*v2s[1]/v2sMagSqr; }

    if(v2aMagSqr==0){ xs2a = xb2;  ys2a = yb2; }
    else{ xs2a = xb2 + (gamma2a-1)*(v2a[0]*xb2 + v2a[1]*yb2 + v2a[2]*zb2)*v2a[1]/v2aMagSqr;  ys2a = yb2 + (gamma2a-1)*(v2a[0]*xb2 + v2a[1]*yb2 + v2a[2]*zb2)*v2a[1]/v2aMagSqr; }



    // Now calculate the distance2 from the string and interpolate the fields

    distance2[0] = sqrt(pow(ys1s - sf_pos1s[0], 2) + pow(zs1s - sf_pos1s[1], 2));
    distance2[1] = sqrt(pow(ys1a - sf_pos1a[0], 2) + pow(zs1a - sf_pos1a[1], 2));
    distance2[2] = sqrt(pow(xs2s - sf_pos2s[0], 2) + pow(ys2s - sf_pos2s[1], 2));
    distance2[3] = sqrt(pow(xs2a - sf_pos2a[0], 2) + pow(ys2a - sf_pos2a[1], 2));

    for (str = 0; str < 4; str++) {

        pClosest[str] = round(distance2[str] / SORa);

        if(pClosest[str]==0){

            // 1st order interpolation since only have grid points on one side.

            phiMag[str] = ( SOR_Fields(pClosest[str]+1,0)*distance2[str] - SOR_Fields(pClosest[str],0)*(distance2[str]-SORa) )/SORa;
            AMag[str] = ( SOR_Fields(pClosest[str]+1,1)*distance2[str] - SOR_Fields(pClosest[str],1)*(distance2[str]-SORa) )/SORa;

        } else if(pClosest[str]<SORnx){

            // 2nd order interpolation

            phiMag[str] = ( SOR_Fields(pClosest[str]-1,0)*(distance2[str]-pClosest[str]*SORa)*(distance2[str]-(pClosest[str]+1)*SORa) -
                            2*SOR_Fields(pClosest[str],0)*(distance2[str]-(pClosest[str]-1)*SORa)*(distance2[str]-(pClosest[str]+1)*SORa) +
                            SOR_Fields(pClosest[str]+1,0)*(distance2[str]-(pClosest[str]-1)*SORa)*(distance2[str]-pClosest[str]*SORa) )/(2*SORa*SORa);

            AMag[str] = ( SOR_Fields(pClosest[str]-1,1)*(distance2[str]-pClosest[str]*SORa)*(distance2[str]-(pClosest[str]+1)*SORa) - 
                            2*SOR_Fields(pClosest[str],1)*(distance2[str]-(pClosest[str]-1)*SORa)*(distance2[str]-(pClosest[str]+1)*SORa) +
                            SOR_Fields(pClosest[str]+1,1)*(distance2[str]-(pClosest[str]-1)*SORa)*(distance2[str]-pClosest[str]*SORa) )/(2*SORa*SORa);

        } else{

            phiMag[str] = 1;

            if(g==0){ AMag[str] = 0; }
            else{ AMag[str] = n/g; }

            // cout << "Off straight string solution grid" << endl;

        }

    }

    // Now need to set phase of phi and split A_theta (theta defined with respect to string) into cartesian components for each string. Modify phase for patching at the same time

    cUnitPos1s = ( -(zs1s - sf_pos1s[1]) + ci*(ys1s - sf_pos1s[0]) )/distance2[0];
    cUnitPos1a = ( (zs1a - sf_pos1a[1]) + ci*(ys1a - sf_pos1a[0]) )/distance2[1];
    cUnitPos2s = ( (xs2s - sf_pos2s[0]) + ci*(ys2s - sf_pos2s[1]) )/distance2[2];
    cUnitPos2a = ( -(xs2a - sf_pos2a[0]) + ci*(ys2a - sf_pos2a[1]) )/distance2[3];


    // Could remove the complex part if I'm not modifying the phase for patching at the same time.

    // string 1

    A1s[0] = 0; // x component is zero

    if(zs1s - sf_pos1s[1] == 0){ phi1s[0] = 0; A1s[1] = 0; } // To prevent division by zero
    else{

        //phi1s[0] = phiMag[0]*real( pow(cUnitPos1s, 0.5*( 1 - tanh(omega*(distance[0] - Lmod)) )) ); // The pow() is the modification for patching
        phi1s[0] = phiMag[0]*real(cUnitPos1s); // Set phase of field without modification. Modification will come later
        A1s[1] = -AMag[0]*(zs1s - sf_pos1s[1])/pow(distance2[0],2); // y component

    }

    if(ys1s - sf_pos1s[0] == 0){ phi1s[1] = 0; A1s[2] = 0; }
    else{

        //phi1s[1] = phiMag[0]*imag( pow(cUnitPos1s, 0.5*( 1 - tanh(omega*(distance[0] - Lmod)) )) );
        phi1s[1] = phiMag[0]*imag(cUnitPos1s);
        A1s[2] = AMag[0]*(ys1s - sf_pos1s[0])/pow(distance2[0],2); // z component

    }

    // antistring 1

    A1a[0] = 0; // x component is zero

    if(zs1a - sf_pos1a[1] == 0){ phi1a[0] = 0; A1a[1] = 0; }
    else{

        //phi1a[0] = phiMag[1]*real( pow(cUnitPos1a, 0.5*( 1 - tanh(omega*(distance[1] - Lmod)) )) );
        phi1a[0] = phiMag[1]*real(cUnitPos1a);
        A1a[1] = AMag[1]*(zs1a - sf_pos1a[1])/pow(distance2[1],2); // y component

    }

    if(ys1a - sf_pos1a[0] == 0){ phi1a[1] = 0; A1a[2] = 0; }
    else{

        //phi1a[1] = phiMag[1]*imag( pow(cUnitPos1a, 0.5*( 1 - tanh(omega*(distance[1] - Lmod)) )) );
        phi1a[1] = phiMag[1]*imag(cUnitPos1a);
        A1a[2] = -AMag[1]*(ys1a - sf_pos1a[0])/pow(distance2[1],2); // z component

    }

    // string 2

    A2s[2] = 0; // z component is zero

    if(xs2s - sf_pos2s[0] == 0){ phi2s[0] = 0; A2s[1] = 0; }
    else{

        //phi2s[0] = phiMag[2]*real( pow(cUnitPos2s, 0.5*( 1 - tanh(omega*(distance[2] - Lmod)) )) );
        phi2s[0] = phiMag[2]*real(cUnitPos2s);
        A2s[1] = AMag[2]*(xs2s - sf_pos2s[0])/pow(distance2[2],2); // y component

    }

    if(ys2s - sf_pos2s[1] == 0){ phi2s[1] = 0; A2s[0] = 0; }
    else{

        //phi2s[1] = phiMag[2]*imag( pow(cUnitPos2s, 0.5*( 1 - tanh(omega*(distance[2] - Lmod)) )) );
        phi2s[1] = phiMag[2]*imag(cUnitPos2s);
        A2s[0] = -AMag[2]*(ys2s - sf_pos2s[1])/pow(distance2[2],2); // x component

    }

    // antistring 2

    A2a[2] = 0; // z component is zero

    if(xs2a - sf_pos2a[0] == 0){ phi2a[0] = 0; A2a[1] = 0; }
    else{

        //phi2a[0] = phiMag[3]*real( pow(cUnitPos2a, 0.5*( 1 - tanh(omega*(distance[3] - Lmod)) )) );
        phi2a[0] = phiMag[3]*real(cUnitPos2a);
        A2a[1] = -AMag[3]*(xs2a - sf_pos2a[0])/pow(distance2[3],2); // y component

    }

    if(ys2a - sf_pos2a[1] == 0){ phi2a[1] = 0; A2a[0] = 0; }
    else{

        //phi2a[1] = phiMag[3]*imag( pow(cUnitPos2a, 0.5*( 1 - tanh(omega*(distance[3] - Lmod)) )) );
        phi2a[1] = phiMag[3]*imag(cUnitPos2a);
        A2a[0] = AMag[3]*(ys2a - sf_pos2a[1])/pow(distance2[3],2); // x component

    }

    // Patch each string - antistring pair together and modify so that the fields are regular at the periodic boundary conditions
    // _0 is the crude patching that doesn't account for the boundary conditions
    // These are the x directed strings





    // -------------------------------------------------------------------------------------------------------------------------------------




    double phi1_0_0 = phi1s[0]*phi1a[0] - phi1s[1]*phi1a[1];
    double phi1_0_1 = phi1s[1]*phi1a[0] + phi1s[0]*phi1a[1];

    // Modify the phase so that the imaginary component is zero at the boundaries and normalise so that the magnitude is unchanged
    // cout << pow(yb2 / (y00 * dy), 2) << endl;
    // cout << pow(zb2 / (z0 * dz), 2) << endl;
    phase_fac = ( 1 - pow(yb2/(y00*dy),2) )*( 1 - pow(zb2/(z0*dz),2) );

    // cout << phase_fac << endl;

    if(pow(phi1_0_0,2) + pow(phi1_0_1*phase_fac,2) == 0){

        phi1_0_0 = phi1_0_1;
        phi1_0_1 = 0;
        

    } else{

        norm_fac = sqrt( (pow(phi1_0_0,2) + pow(phi1_0_1,2))/(pow(phi1_0_0,2) + pow(phi1_0_1*phase_fac,2)) );
        phi1_0_0 = norm_fac*phi1_0_0;
        phi1_0_1 = norm_fac*phase_fac*phi1_0_1;
        
    }
    
    
    // Do the same for the second string - antistring pair (z directed)

    double phi2_0_0 = phi2s[0]*phi2a[0] - phi2s[1]*phi2a[1];
    double phi2_0_1 = phi2s[1]*phi2a[0] + phi2s[0]*phi2a[1];

    phase_fac = ( 1 - pow(xb2/(x0*dx),2) )*( 1 - pow(yb2/(y00*dy),2) );

    if(pow(phi2_0_0,2) + pow(phi2_0_1*phase_fac,2) == 0){

        phi2_0_0 = phi2_0_1;
        phi2_0_1 = 0;

    } else{
    
        norm_fac = sqrt( (pow(phi2_0_0,2) + pow(phi2_0_1,2))/(pow(phi2_0_0,2) + pow(phi2_0_1*phase_fac,2)) );
        phi2_0_0 = norm_fac*phi2_0_0;
        phi2_0_1 = norm_fac*phase_fac*phi2_0_1;

    }


    // At is needed for the gauge transformation at the next time step. Gauge transformation sets At(t=0) to zero without affecting Ai(t=0) but need to calculate as it
    // will have an effect on Ai(t=dt). Non-zero At is created by the Lorentz transformation.

    Array X(4, 0.0);
    X(0) = phi1_0_0;
    X(1) = phi1_0_1;
    X(2) = phi2_0_0;
    X(3) = phi2_0_1;


    return X;
}
Array ProcessLooper(double xb1, double yb1, double zb1, int i, int j, int k) {

    int condition0_1 = ny - 1;
    int condition1_1 = nx - 1;
    
    double condition1 = (condition0_1 - y00) * dy;
    double condition2 = (0 - y00) * dy;
    double condition3 = (condition1_1 - x0) * dx;
    double condition4 = (0 - x0) * dx;

    Array X_1 = ProcessLoop(xb1, condition1, zb1, i, condition0_1, k);
    Array X_2 = ProcessLoop(xb1, condition2, zb1, i, 0, k);
    Array X_3 = ProcessLoop(condition3, yb1, zb1, condition1_1, j, k);
    Array X_4 = ProcessLoop(condition4, yb1, zb1, 0, j, k);

    phi1Intrp = (sqrt(pow(X_1(0), 2) + pow(X_1(1), 2)) - sqrt(pow(X_2(0), 2) + pow(X_2(1), 2))) * (yb1 + y00 * dy) / (2 * y00 * dy)
        + sqrt(pow(X_2(0), 2) + pow(X_2(1), 2));

    phi2Intrp = (sqrt(pow(X_3(2), 2) + pow(X_3(3), 2)) - sqrt(pow(X_4(2), 2) + pow(X_4(3), 2))) * (xb1 + x0 * dx) / (2 * x0 * dx)
        + sqrt(pow(X_4(2), 2) + pow(X_4(3), 2));

    Array X = ProcessLoop(xb1, yb1, zb1, i, j, k);

    double phi1_1_0 = X(0) / phi1Intrp;
    double phi1_1_1 = X(1) / phi1Intrp;

    double phi2_1_0 = X(2) / phi2Intrp;
    double phi2_1_1 = X(3) / phi2Intrp;
    Array Z(4, 0.0);
    Z(0) = phi1_1_0;
    Z(1) = phi1_1_1;
    Z(2) = phi2_1_0;
    Z(3) = phi2_1_1;

    

    // cout << X_3(3) << endl;
    return Z;
}

Array ProcessLoop2(double xb3, double yb3, double yz3, int i, int j, int k) {
    // Firstly need to calculate what these positions correspond to in the frame with a stationary string at t=0.

    v1sMagSqr = pow(v1s[0],2) + pow(v1s[1],2) + pow(v1s[2],2);
    v1aMagSqr = pow(v1a[0],2) + pow(v1a[1],2) + pow(v1a[2],2);
    v2sMagSqr = pow(v2s[0],2) + pow(v2s[1],2) + pow(v2s[2],2);
    v2aMagSqr = pow(v2a[0],2) + pow(v2a[1],2) + pow(v2a[2],2);

    gamma1s = 1/sqrt(1 - v1sMagSqr);
    gamma1a = 1/sqrt(1 - v1aMagSqr);
    gamma2s = 1/sqrt(1 - v2sMagSqr);
    gamma2a = 1/sqrt(1 - v2aMagSqr);

    if(v1sMagSqr==0){ sf_pos1s[0] = pos1s[0];  sf_pos1s[1] = pos1s[1]; }
    else{ sf_pos1s[0] = pos1s[0] + (gamma1s-1)*(v1s[1]*pos1s[0] + v1s[2]*pos1s[1])*v1s[1]/v1sMagSqr;  sf_pos1s[1] = pos1s[1] + (gamma1s-1)*(v1s[1]*pos1s[0] + v1s[2]*pos1s[1])*v1s[2]/v1sMagSqr; }

    if(v1aMagSqr==0){ sf_pos1a[0] = pos1a[0];  sf_pos1a[1] = pos1a[1]; }
    else{ sf_pos1a[0] = pos1a[0] + (gamma1a-1)*(v1a[1]*pos1a[0] + v1a[2]*pos1a[1])*v1a[1]/v1aMagSqr;  sf_pos1a[1] = pos1a[1] + (gamma1a-1)*(v1a[1]*pos1a[0] + v1a[2]*pos1a[1])*v1a[2]/v1aMagSqr; }

    if(v2sMagSqr==0){ sf_pos2s[0] = pos2s[0];  sf_pos2s[1] = pos2s[1]; }
    else{ sf_pos2s[0] = pos2s[0] + (gamma2s-1)*(v2s[0]*pos2s[0] + v2s[1]*pos2s[1])*v2s[0]/v2sMagSqr;  sf_pos2s[1] = pos2s[1] + (gamma2s-1)*(v2s[0]*pos2s[0] + v2s[1]*pos2s[1])*v2s[1]/v2sMagSqr; }

    if(v2aMagSqr==0){ sf_pos2a[0] = pos2a[0];  sf_pos2a[1] = pos2a[1]; }
    else{ sf_pos2a[0] = pos2a[0] + (gamma2a-1)*(v2a[0]*pos2a[0] + v2a[1]*pos2a[1])*v2a[0]/v2aMagSqr;  sf_pos2a[1] = pos2a[1] + (gamma2a-1)*(v2a[0]*pos2a[0] + v2a[1]*pos2a[1])*v2a[1]/v2aMagSqr; }




    if (v1sMagSqr == 0) { ys1s = yb3;  zs1s = yz3; }
    else { ys1s = yb3 + (gamma1s - 1) * (v1s[0] * xb3 + v1s[1] * yb3 + v1s[2] * yz3) * v1s[1] / v1sMagSqr - gamma1s*v1s[1]*dt;
           zs1s = yz3 + (gamma1s - 1) * (v1s[0] * xb3 + v1s[1] * yb3 + v1s[2] * yz3) * v1s[2] / v1sMagSqr - gamma1s*v1s[2]*dt; }

    if (v1aMagSqr == 0) { ys1a = yb3;  zs1a = yz3; }
    else { ys1a = yb3 + (gamma1a - 1) * (v1a[0] * xb3 + v1a[1] * yb3 + v1a[2] * yz3) * v1a[1] / v1aMagSqr - gamma1a*v1a[1]*dt;
           zs1a = yz3 + (gamma1a - 1) * (v1a[0] * xb3 + v1a[1] * yb3 + v1a[2] * yz3) * v1a[2] / v1aMagSqr - gamma1a*v1a[2]*dt; }

    if (v2sMagSqr == 0) { xs2s = xb3;  ys2s = yb3; }
    else { xs2s = xb3 + (gamma2s - 1) * (v2s[0] * xb3 + v2s[1] * yb3 + v2s[2] * yz3) * v2s[0] / v2sMagSqr - gamma2s*v2s[0]*dt;
           ys2s = yb3 + (gamma2s - 1) * (v2s[0] * xb3 + v2s[1] * yb3 + v2s[2] * yz3) * v2s[1] / v2sMagSqr - gamma2s*v2s[1]*dt; }

    if (v2aMagSqr == 0) { xs2a = xb3;  ys2a = yb3; }
    else { xs2a = xb3 + (gamma2a - 1) * (v2a[0] * xb3 + v2a[1] * yb3 + v2a[2] * yz3) * v2a[1] / v2aMagSqr - gamma2a*v2a[0]*dt;
           ys2a = yb3 + (gamma2a - 1) * (v2a[0] * xb3 + v2a[1] * yb3 + v2a[2] * yz3) * v2a[1] / v2aMagSqr - gamma2a*v2a[1]*dt;; }


    // Now calculate the distance2 from the string and interpolate the fields

    distance2[0] = sqrt(pow(ys1s - sf_pos1s[0],2) + pow(zs1s - sf_pos1s[1],2));
    distance2[1] = sqrt(pow(ys1a - sf_pos1a[0],2) + pow(zs1a - sf_pos1a[1],2)); 
    distance2[2] = sqrt(pow(xs2s - sf_pos2s[0],2) + pow(ys2s - sf_pos2s[1],2)); 
    distance2[3] = sqrt(pow(xs2a - sf_pos2a[0],2) + pow(ys2a - sf_pos2a[1],2));

    for (str = 0; str < 4; str++) {

        pClosest[str] = round(distance2[str] / SORa);

        if (pClosest[str] == 0) {

            // 1st order interpolation since only have grid points on one side.

            phiMag[str] = (SOR_Fields(pClosest[str] + 1, 0) * distance2[str] - SOR_Fields(pClosest[str], 0) * (distance2[str] - SORa)) / SORa;
            AMag[str] = (SOR_Fields(pClosest[str] + 1, 1) * distance2[str] - SOR_Fields(pClosest[str], 1) * (distance2[str] - SORa)) / SORa;

        }
        else if (pClosest[str] < SORnx) {

            // 2nd order interpolation

            phiMag[str] = (SOR_Fields(pClosest[str] - 1, 0) * (distance2[str] - pClosest[str] * SORa) * (distance2[str] - (pClosest[str] + 1) * SORa) -
                2 * SOR_Fields(pClosest[str], 0) * (distance2[str] - (pClosest[str] - 1) * SORa) * (distance2[str] - (pClosest[str] + 1) * SORa) +
                SOR_Fields(pClosest[str] + 1, 0) * (distance2[str] - (pClosest[str] - 1) * SORa) * (distance2[str] - pClosest[str] * SORa)) / (2 * SORa * SORa);

            AMag[str] = (SOR_Fields(pClosest[str] - 1, 1) * (distance2[str] - pClosest[str] * SORa) * (distance2[str] - (pClosest[str] + 1) * SORa) -
                2 * SOR_Fields(pClosest[str], 1) * (distance2[str] - (pClosest[str] - 1) * SORa) * (distance2[str] - (pClosest[str] + 1) * SORa) +
                SOR_Fields(pClosest[str] + 1, 1) * (distance2[str] - (pClosest[str] - 1) * SORa) * (distance2[str] - pClosest[str] * SORa)) / (2 * SORa * SORa);

        }
        else{

            phiMag[str] = 1;

            if(g==0){ AMag[str] = 0; }
            else{ AMag[str] = n/g; }

            // cout << "Off straight string solution grid" << endl;

        }

    }

    // Now need to set phase of phi and split A_theta (theta defined with respect to string) into cartesian components for each string. Modify phase for patching at the same time

    cUnitPos1s = (-(zs1s - sf_pos1s[1]) + ci * (ys1s - sf_pos1s[0])) / distance2[0];
    cUnitPos1a = ((zs1a - sf_pos1a[1]) + ci * (ys1a - sf_pos1a[0])) / distance2[1];
    cUnitPos2s = ((xs2s - sf_pos2s[0]) + ci * (ys2s - sf_pos2s[1])) / distance2[2];
    cUnitPos2a = (-(xs2a - sf_pos2a[0]) + ci * (ys2a - sf_pos2a[1])) / distance2[3];


    // Could remove the complex part if I'm not modifying the phase for patching at the same time.

    // string 1

    // A1s[0] = 0; // x component is zero

    if (zs1s - sf_pos1s[1] == 0) { phi1s[0] = 0; A1s[1] = 0; } // To prevent division by zero
    else {

        //phi1s[0] = phiMag[0]*real( pow(cUnitPos1s, 0.5*( 1 - tanh(omega*(distance2[0] - Lmod)) )) ); // The pow() is the modification for patching
        phi1s[0] = phiMag[0] * real(cUnitPos1s); // Set phase of field without modification. Modification will come later
        A1s[1] = -AMag[0] * (zs1s - sf_pos1s[1]) / pow(distance2[0], 2); // y component

    }

    if (ys1s - sf_pos1s[0] == 0) { phi1s[1] = 0; A1s[2] = 0; }
    else {

        //phi1s[1] = phiMag[0]*imag( pow(cUnitPos1s, 0.5*( 1 - tanh(omega*(distance2[0] - Lmod)) )) );
        phi1s[1] = phiMag[0] * imag(cUnitPos1s);
        A1s[2] = AMag[0] * (ys1s - sf_pos1s[0]) / pow(distance2[0], 2); // z component

    }

    // antistring 1

    A1a[0] = 0; // x component is zero

    if (zs1a - sf_pos1a[1] == 0) { phi1a[0] = 0; A1a[1] = 0; }
    else {

        //phi1a[0] = phiMag[1]*real( pow(cUnitPos1a, 0.5*( 1 - tanh(omega*(distance2[1] - Lmod)) )) );
        phi1a[0] = phiMag[1] * real(cUnitPos1a);
        A1a[1] = AMag[1] * (zs1a - sf_pos1a[1]) / pow(distance2[1], 2); // y component

    }

    if (ys1a - sf_pos1a[0] == 0) { phi1a[1] = 0; A1a[2] = 0; }
    else {

        //phi1a[1] = phiMag[1]*imag( pow(cUnitPos1a, 0.5*( 1 - tanh(omega*(distance2[1] - Lmod)) )) );
        phi1a[1] = phiMag[1] * imag(cUnitPos1a);
        A1a[2] = -AMag[1] * (ys1a - sf_pos1a[0]) / pow(distance2[1], 2); // z component

    }

    // string 2

    // A2s[2] = 0; // z component is zero

    if (xs2s - sf_pos2s[0] == 0) { phi2s[0] = 0; A2s[1] = 0; }
    else {

        //phi2s[0] = phiMag[2]*real( pow(cUnitPos2s, 0.5*( 1 - tanh(omega*(distance2[2] - Lmod)) )) );
        phi2s[0] = phiMag[2] * real(cUnitPos2s);
        A2s[1] = AMag[2] * (xs2s - sf_pos2s[0]) / pow(distance2[2], 2); // y component

    }

    if (ys2s - sf_pos2s[1] == 0) { phi2s[1] = 0; A2s[0] = 0; }
    else {

        //phi2s[1] = phiMag[2]*imag( pow(cUnitPos2s, 0.5*( 1 - tanh(omega*(distance2[2] - Lmod)) )) );
        phi2s[1] = phiMag[2] * imag(cUnitPos2s);
        A2s[0] = -AMag[2] * (ys2s - sf_pos2s[1]) / pow(distance2[2], 2); // x component

    }

    // antistring 2

    A2a[2] = 0; // z component is zero

    if (xs2a - sf_pos2a[0] == 0) { phi2a[0] = 0; A2a[1] = 0; }
    else {

        //phi2a[0] = phiMag[3]*real( pow(cUnitPos2a, 0.5*( 1 - tanh(omega*(distance2[3] - Lmod)) )) );
        phi2a[0] = phiMag[3] * real(cUnitPos2a);
        A2a[1] = -AMag[3] * (xs2a - sf_pos2a[0]) / pow(distance2[3], 2); // y component

    }

    if (ys2a - sf_pos2a[1] == 0) { phi2a[1] = 0; A2a[0] = 0; }
    else {

        //phi2a[1] = phiMag[3]*imag( pow(cUnitPos2a, 0.5*( 1 - tanh(omega*(distance2[3] - Lmod)) )) );
        phi2a[1] = phiMag[3] * imag(cUnitPos2a);
        A2a[0] = AMag[3] * (ys2a - sf_pos2a[1]) / pow(distance2[3], 2); // x component

    }

    // Patch each string - antistring pair together and modify so that the fields are regular at the periodic boundary conditions
    // _0 is the crude patching that doesn't account for the boundary conditions
    // These are the x directed strings





    // -------------------------------------------------------------------------------------------------------------------------------------




    double phi1_0_0 = phi1s[0] * phi1a[0] - phi1s[1] * phi1a[1];
    double phi1_0_1 = phi1s[1] * phi1a[0] + phi1s[0] * phi1a[1];

    // Modify the phase so that the imaginary component is zero at the boundaries and normalise so that the magnitude is unchanged
    // cout << pow(yb3 / (y00 * dy), 2) << endl;
    // cout << pow(yz3 / (z0 * dz), 2) << endl;
    phase_fac = (1 - pow(yb3 / (y00 * dy), 2)) * (1 - pow(yz3 / (z0 * dz), 2));

    if (pow(phi1_0_0, 2) + pow(phi1_0_1 * phase_fac, 2) == 0) {

        phi1_0_0 = phi1_0_1;
        phi1_0_1 = 0;

    }
    else {

        norm_fac = sqrt((pow(phi1_0_0, 2) + pow(phi1_0_1, 2)) / (pow(phi1_0_0, 2) + pow(phi1_0_1 * phase_fac, 2)));
        phi1_0_0 = norm_fac * phi1_0_0;
        phi1_0_1 = norm_fac * phase_fac * phi1_0_1;

    }


    // Do the same for the second string - antistring pair (z directed)

    double phi2_0_0 = phi2s[0] * phi2a[0] - phi2s[1] * phi2a[1];
    double phi2_0_1 = phi2s[1] * phi2a[0] + phi2s[0] * phi2a[1];

    phase_fac = (1 - pow(xb3 / (x0 * dx), 2)) * (1 - pow(yb3 / (y00 * dy), 2));

    if (pow(phi2_0_0, 2) + pow(phi2_0_1 * phase_fac, 2) == 0) {

        phi2_0_0 = phi2_0_1;
        phi2_0_1 = 0;

    }
    else {

        norm_fac = sqrt((pow(phi2_0_0, 2) + pow(phi2_0_1, 2)) / (pow(phi2_0_0, 2) + pow(phi2_0_1 * phase_fac, 2)));
        phi2_0_0 = norm_fac * phi2_0_0;
        phi2_0_1 = norm_fac * phase_fac * phi2_0_1;

    }

    // At is needed for the gauge transformation at the next time step. Gauge transformation sets At(t=0) to zero without affecting Ai(t=0) but need to calculate as it
    // will have an effect on Ai(t=dt). Non-zero At is created by the Lorentz transformation.

    Array X(4, 0.0);
    X(0) = phi1_0_0;
    X(1) = phi1_0_1;
    X(2) = phi2_0_0;
    X(3) = phi2_0_1;

    return X;
}
Array ProcessLooper2(double xb4, double yb4, double zb4, int i, int j, int k) {

    int condition0_1 = ny - 1;
    int condition1_1 = nx - 1;
    
    double condition1 = (condition0_1 - y00) * dy;
    double condition2 = (0 - y00) * dy;
    double condition3 = (condition1_1 - x0) * dx;
    double condition4 = (0 - x0) * dx;

    Array X_1 = ProcessLoop2(xb4, condition1, zb4, i, condition0_1, k);
    Array X_2 = ProcessLoop2(xb4, condition2, zb4, i, 0, k);
    Array X_3 = ProcessLoop2(condition3, yb4, zb4, condition1_1, j, k);
    Array X_4 = ProcessLoop2(condition4, yb4, zb4, 0, j, k);

    phi1Intrp = (sqrt(pow(X_1(0), 2) + pow(X_1(1), 2)) - sqrt(pow(X_2(0), 2) + pow(X_2(1), 2))) * (yb4 + y00 * dy) / (2 * y00 * dy)
        + sqrt(pow(X_2(0), 2) + pow(X_2(1), 2));

    phi2Intrp = (sqrt(pow(X_3(2), 2) + pow(X_3(3), 2)) - sqrt(pow(X_4(2), 2) + pow(X_4(3), 2))) * (xb4 + x0 * dx) / (2 * x0 * dx)
        + sqrt(pow(X_4(2), 2) + pow(X_4(3), 2));

    Array X = ProcessLoop2(xb4, yb4, zb4, i, j, k);

    double phi1_1_0 = X(0) / phi1Intrp;
    double phi1_1_1 = X(1) / phi1Intrp;

    double phi2_1_0 = X(2) / phi2Intrp;
    double phi2_1_1 = X(3) / phi2Intrp;
    Array Z(4, 0.0);
    Z(0) = phi1_1_0;
    Z(1) = phi1_1_1;
    Z(2) = phi2_1_0;
    Z(3) = phi2_1_1;

    
    // cout << X_3(3) << endl;
    return Z;
}


int main() {
    //lambda = dz*(nz-1)/(1-0.25*eps*eps);
    for (int i = 0; i < SORnx; i++) {

        SOR_input >> SOR_Fields(i, 0) >> SOR_Fields(i, 1);

    }
    //Array phi(2, nx, ny, nz, 0.0), A(3, nx, ny, nz, 0.0);
    double xb, yb, zb;
    int i, j, k, comp;


    string file_path = __FILE__;

    string icPath = dir_path + "/Data/ic.txt";
    string test1sPath = dir_path + "/Data/test1s.txt";
    string test1aPath = dir_path + "/Data/test1a.txt";
    string test2sPath = dir_path + "/Data/test2s.txt";
    string test2aPath = dir_path + "/Data/test2a.txt";

    ofstream ic(icPath.c_str());
    ofstream test1s(test1sPath.c_str());
    ofstream test1a(test1aPath.c_str());
    ofstream test2s(test2sPath.c_str());
    ofstream test2a(test2aPath.c_str());

    if (ic_type == "loop collision") {

        // Could probably simplify a lot of this by incorporating all strings into one array. i.e make At(4,nx,ny,nz,0.0)
 


        // Grid positions of zero.


        if(v1sMagSqr>=1 or v1aMagSqr>=1 or v2sMagSqr>=1 or v2aMagSqr>=1){ cout << "Error: boost velocity is greater than the speed of light" << endl; }



        // Calculate what the positions of the strings corresponds to in their stationary frames
        // #pragma omp parallel for default(none) private(j,k,xb,yb,zb,phi1Intrp,ys1s,zs1s,ys1a,zs1a,xs2s,ys2s,xs2a,ys2a,distance2,pClosest,phiMag,AMag,cUnitPos1s,cUnitPos1a,cUnitPos2s,cUnitPos2a, A1s,phi1s, \
        // A1a,phi1a,A2s,phi2s,A2a,phi2a,phase_fac,norm_fac,str,comp,A1sPatch,A1aPatch,A2sPatch,A2aPatch,phi1Intrp, phi2Intrp) shared(v1sMagSqr,v1aMagSqr,v2sMagSqr,v2aMagSqr,gamma1s,gamma1a,gamma2s,gamma2a,x0,y00,z0,v1s,v1a,v2s,v2a,sf_pos1s,sf_pos1a,sf_pos2s,sf_pos2a,SOR_Fields,ci)

        // Apply first set of interpolation so that magnitude of fields are continuous across the periodic boundary conditions

        for(i=0;i<nx;i++){

            xb = (i-x0)*dx;

            for(j=0;j<ny;j++){

                yb = (j-y00)*dy;

                for(k=0;k<nz;k++){

                    zb = (k-z0)*dy;

                    // Array L = ProcessLoop2(xb, yb, zb, i, j, k);
                    
                    Array Z = ProcessLooper(xb, yb, zb, i, j, k);
                  
                    // Define the interpolation functions

                    Array Z_1 = ProcessLooper(xb, yb, (nz - 1 - z0) * dz, i, j, nz-1);
                    Array Z_2 = ProcessLooper(xb, yb, (0 - z0) * dz, i, j, 0);
                    Array Z_3 = ProcessLooper(xb, (ny - 1 - y00) * dy, zb, i, ny-1, k);
                    Array Z_4 = ProcessLooper(xb, (0 - y00) * dy, zb, i, 0, k);

                    phi1Intrp = (sqrt(pow(Z_1(0), 2) + pow(Z_1(1), 2)) - sqrt(pow(Z_2(0), 2) + pow(Z_2(1), 2))) * (zb + z0 * dz) / (2 * z0 * dz)
                        + sqrt(pow(Z_2(0), 2) + pow(Z_2(1), 2));

                    phi2Intrp = (sqrt(pow(Z_3(2), 2) + pow(Z_3(3), 2)) - sqrt(pow(Z_4(2), 2) + pow(Z_4(3), 2))) * (yb + y00 * dy) / (2 * y00 * dy)
                        + sqrt(pow(Z_4(2), 2) + pow(Z_4(3), 2));



                    phi1_2[0] = Z(0) / phi1Intrp;
                    phi1_2[1] = Z(1) / phi1Intrp;

                    phi2_2[0] = Z(2) / phi2Intrp;
                    phi2_2[1] = Z(3) / phi2Intrp;
                    
                    int zero_value = 0;

                    // Output the field values at the initial time.
                    ic << (phi1_2[0] * phi2_2[0] - phi1_2[1] * phi2_2[1]) << " " << (phi1_2[0] * phi2_2[1] + phi1_2[1] * phi2_2[0]) << " " << zero_value << " " << zero_value << " " << zero_value << endl;

                }

            }

        }

        // 2nd time step:

        for(i=0;i<nx;i++){

            xb = (i-x0)*dx;
            ip = (i+1+nx)%nx;
            im = (i-1+nx)%nx;

            for(j=0;j<ny;j++){

                yb = (j-y00)*dy;
                jp = (j+1+ny)%ny;
                jm = (j-1+ny)%ny;

                for(k=0;k<nz;k++){

                    zb = (k-z0)*dy;
                    kp = (k+1+nz)%nz;
                    km = (k-1+nz)%nz;

                    Array Z = ProcessLooper2(xb, yb, zb, i, j, k);
                    // Define the interpolation functions
                    
                    // Define the interpolation functions
                    Array Z_1 = ProcessLooper2(xb, yb, (nz - 1 - z0) * dz, i, j, nz-1);
                    Array Z_2 = ProcessLooper2(xb, yb, (0 - z0) * dz, i, j, 0);
                    Array Z_3 = ProcessLooper2(xb, (ny - 1 - y00) * dy, zb, i, ny-1, k);
                    Array Z_4 = ProcessLooper2(xb, ((0 - y00) * dy), zb, i, 0, k);

                    phi1Intrp = (sqrt(pow(Z_1(0), 2) + pow(Z_1(1), 2)) - sqrt(pow(Z_2(0), 2) + pow(Z_2(1), 2))) * (zb + z0 * dz) / (2 * z0 * dz)
                        + sqrt(pow(Z_2(0), 2) + pow(Z_2(1), 2));

                    phi2Intrp = (sqrt(pow(Z_3(2), 2) + pow(Z_3(3), 2)) - sqrt(pow(Z_4(2), 2) + pow(Z_4(3), 2))) * (yb + y00 * dy) / (2 * y00 * dy)
                        + sqrt(pow(Z_4(2), 2) + pow(Z_4(3), 2));


                    phi1_2[0] = Z(0) / phi1Intrp;
                    phi1_2[1] = Z(1) / phi1Intrp;

                    phi2_2[0] = Z(2) / phi2Intrp;
                    phi2_2[1] = Z(3) / phi2Intrp;

                    int zero_value = 0;

                    // Output the field values at the initial time.
                    
                    ic << (phi1_2[0] * phi2_2[0] - phi1_2[1] * phi2_2[1]) << " " << (phi1_2[0] * phi2_2[1] + phi1_2[1] * phi2_2[0]) << " " << zero_value << " " << zero_value << " " << zero_value << endl;

                }

            }

        }

    }
    
    else {

        cout << "Unrecognised ic_type" << endl;

    }


    return 0;

}