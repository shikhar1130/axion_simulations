#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <complex>
#include <random>
#include <limits> 
#include <cmath>
#include "array.hpp"

using namespace std;
typedef complex<double> dcmplx;

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
double distance3D(double x1, double y1, double z1, double x2, double y2, double z2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<double> x_string(double sigma, double loop_radius) {
    std::vector<double> position(3); // Initialize a 3D vector

    // Define the position as a function of sigma
    // Here we use simple trigonometric functions as an example
    position[0] = loop_radius * cos(2 * M_PI * sigma); // x-coordinate
    position[1] = loop_radius * sin(2 * M_PI * sigma); // y-coordinate
    position[2] = 0; // z-coordinate, currently set to 0

    return position;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<double> x_string_derivative(double sigma, double loop_radius) {
    std::vector<double> position(3); // Initialize a 3D vector

    // Define the position as a function of sigma
    // Here we use simple trigonometric functions as an example
    position[0] = - 2 * M_PI * loop_radius * sin(2 * M_PI * sigma); // x-coordinate
    position[1] = 2 * M_PI * loop_radius * cos(2 * M_PI * sigma); // y-coordinate
    position[2] = 0; // z-coordinate, currently set to 0

    return position;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<double> x_string_dt(double sigma, double loop_radius, double velocity, double dt) {
    std::vector<double> position(3); // Initialize a 3D vector

    // Define the position as a function of sigma
    // Here we use simple trigonometric functions as an example
    position[0] = loop_radius * cos(2 * M_PI * sigma) * cos(velocity * dt); // x-coordinate
    position[1] = loop_radius * sin(2 * M_PI * sigma); // y-coordinate
    position[2] = loop_radius * cos(2 * M_PI * sigma) * sin(velocity * dt); // z-coordinate, currently set to 0

    return position;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
std::vector<double> x_string_dt_derivative(double sigma, double loop_radius, double velocity, double dt) {
    std::vector<double> position(3); // Initialize a 3D vector

    // Define the position as a function of sigma
    // Here we use simple trigonometric functions as an example
    position[0] = - 2 * M_PI * loop_radius * sin(2 * M_PI * sigma) * cos(velocity * dt); // x-coordinate
    position[1] = 2 * M_PI * loop_radius * cos(2 * M_PI * sigma); // y-coordinate
    position[2] = - 2 * M_PI * loop_radius * sin(2 * M_PI * sigma) * sin(velocity * dt); // z-coordinate, currently set to 0

    return position;
}

// Perturb a straight, global string solution, evolve the system and investigate how it radiates.


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                 		  Parameters & Declarations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const string ic_type = "loop";	 // Which type of initial condition generation to use. "NG sine" bases initial conditions on the Nambu-Goto sine wave solution constructed with straight string solutions 
                                 	 // "simple sine" offsets the straight string solutions (x position) by a sine wave
			         	 // "random" creates random initial conditions. "boost" creates a single straight (z directed) string with a Lorentz boost applied.
                                 	 // "loop collision" creates two sets of (seperated) string, anti-string pairs. They are boosted towards each other and patched together so that they will collide
                                 	 // and form a loop (2 one due to periodic boundary conditions which are required for this sim) 


const int nx = 1024;
const int ny = 1024;
const int nz = 1024;
const double dx = 0.7;
const double dy = 0.7;
const double dz = 0.7;
const double dt = 0.14;

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

// For the single loop case //////////////////////////////////////////////////////////
const double loop_radius = 0.25*(nx-1)*dx; // Define the radius of the loop
const int num_points = 1000;
double step = 1.0 / num_points;

const double velocity = 1 / (2 * loop_radius); // Define the velocity of the loop

std::vector<double> sigma(num_points + 1);

int main() {

    string dir_path = "/home/agarwal/1024_v_2R_0.7";
    string icPath = dir_path + "/Data/ic.txt";
    ofstream ic (icPath.c_str());

    if (ic_type == "loop") {
        Array SOR_Fields(SORnx, 2, 0.0);
        double distance, phiMag, AMag, xb, yb, zb;
        int pClosest, i, j, k;

        string SOR_inputPath = dir_path + "/Data/SOR_Fields.txt";

        ifstream SOR_input (SOR_inputPath.c_str());

        for(i=0;i<SORnx;i++){

           SOR_input >> SOR_Fields(i,0) >> SOR_Fields(i,1);

        }

        for (int i = 0; i <= num_points; ++i) {
            sigma[i] = i * step;
        }

        int x0 = round((nx-1)/2);
        int y0 = round((ny-1)/2);
        int z0 = round((nz-1)/2);

        // Loop over grid points
        for(i=0;i<nx;i++){

            xb = (i-x0)*dx;

            for(j=0;j<ny;j++){

                yb = (j-y0)*dy;

                for(k=0;k<nz;k++){

                    zb = (k-z0)*dy;

                    double distance = numeric_limits<double>::max();
                    double phi[2];
                    double closest_s = 0; // Closest point on loop to grid point

                    double s = std::atan2(yb, xb) / (2 * M_PI);

                    // Calculate position on loop for the current sigma value
                    std::vector<double> loop_pos = x_string(s, loop_radius);

                    double dist = distance3D(xb, yb, zb, loop_pos[0], loop_pos[1], loop_pos[2]);
                    distance = dist;


                    pClosest = round(distance / SORa);

                    if (pClosest == 0) {
                        
                        phiMag = (SOR_Fields(pClosest + 1, 0) * distance - SOR_Fields(pClosest, 0) * (distance - SORa)) / SORa;
                        AMag = (SOR_Fields(pClosest + 1, 1) * distance - SOR_Fields(pClosest, 1) * (distance - SORa)) / SORa;
                    
                    } else if (pClosest < SORnx) {

                        phiMag = (SOR_Fields(pClosest - 1, 0) * (distance - pClosest * SORa) * (distance - (pClosest + 1) * SORa) -
                                  2 * SOR_Fields(pClosest, 0) * (distance - (pClosest - 1) * SORa) * (distance - (pClosest + 1) * SORa) +
                                  SOR_Fields(pClosest + 1, 0) * (distance - (pClosest - 1) * SORa) * (distance - pClosest * SORa)) / (2 * SORa * SORa);
                        AMag = (SOR_Fields(pClosest - 1, 1) * (distance - pClosest * SORa) * (distance - (pClosest + 1) * SORa) -
                                  2 * SOR_Fields(pClosest, 1) * (distance - (pClosest - 1) * SORa) * (distance - (pClosest + 1) * SORa) +
                                  SOR_Fields(pClosest + 1, 1) * (distance - (pClosest - 1) * SORa) * (distance - pClosest * SORa)) / (2 * SORa * SORa);
                    
                    } else {

                        phiMag = 1;
                        if (g == 0) {
                            AMag = 0;
                        } else {
                            AMag = n / g;
                        }
                        // cout << "Off straight string solution grid" << endl;
                    }

                    std::vector<double> nearest_loop_pos = x_string(s, loop_radius);
                    std::vector<double> s_m(3);
                    std::vector<double> c(3);  
                    std::vector<double> derivative(3);
                    std::vector<double> cross_product(3);

                    s_m[0] = xb - nearest_loop_pos[0];
                    s_m[1] = yb - nearest_loop_pos[1];
                    s_m[2] = zb - nearest_loop_pos[2];

                    // Calculate the magnitude of s_m
                    double s_m_mag = sqrt(pow(s_m[0], 2) + pow(s_m[1], 2) + pow(s_m[2], 2));

                    // Normalize s_m to get its unit vector

                    if (s_m_mag != 0){
                        s_m[0] /= s_m_mag; 
                        s_m[1] /= s_m_mag;
                        s_m[2] /= s_m_mag;

                        // Calculate the magnitude of nearest_loop_pos to normalize it
                        double nearest_loop_pos_mag = sqrt(pow(nearest_loop_pos[0], 2) + pow(nearest_loop_pos[1], 2) + pow(nearest_loop_pos[2], 2));

                        // Calculate the unit vector for nearest_loop_pos
                        c[0] = nearest_loop_pos[0] / nearest_loop_pos_mag;
                        c[1] = nearest_loop_pos[1] / nearest_loop_pos_mag;
                        c[2] = nearest_loop_pos[2] / nearest_loop_pos_mag;

                        double dot_product = (c[0] * s_m[0]) + (c[1] * s_m[1]) + (c[2] * s_m[2]);

                        //For imaginary part
                        cross_product[0] = (s_m[1] * c[2]) - (s_m[2] * c[1]);
                        cross_product[1] = (s_m[2] * c[0]) - (s_m[0] * c[2]);
                        cross_product[2] = (s_m[0] * c[1]) - (s_m[1] * c[0]);
                        
                        derivative = x_string_derivative(s, loop_radius);
                        double derivative_mag = sqrt(pow(derivative[0], 2) + pow(derivative[1], 2) + pow(derivative[2], 2));

                        derivative[0] /= derivative_mag;
                        derivative[1] /= derivative_mag;
                        derivative[2] /= derivative_mag;

                        double imag = cross_product[0] * derivative[0] + cross_product[1] * derivative[1] + cross_product[2] * derivative[2];
                        
                        phi[0] = phiMag * dot_product;
                        phi[1] = phiMag * imag;

                    }
                    else{
                        phi[0] = 0;
                        phi[1] = 0;
                    }
                    int zero_value = 0;

                    ic << phi[0] << " " << phi[1] << " " << zero_value << " " << zero_value << " " << zero_value << endl;

                }
            }
        }

        // Second time step
        for(i=0;i<nx;i++){

            xb = (i-x0)*dx;

            for(j=0;j<ny;j++){

                yb = (j-y0)*dy;

                for(k=0;k<nz;k++){

                    zb = (k-z0)*dy;

                    double distance = numeric_limits<double>::max();
                    double phi[2];
                    double closest_s = 0; // Closest point on loop to grid point

                    double s = std::atan2(yb, (xb*cos(velocity * dt) + zb * sin(velocity * dt))) / (2 * M_PI);

                    // Calculate position on loop for the current sigma value
                    std::vector<double> loop_pos = x_string_dt(s, loop_radius, velocity, dt);

                    double dist = distance3D(xb, yb, zb, loop_pos[0], loop_pos[1], loop_pos[2]);
                    distance = dist;

                    pClosest = round(distance / SORa);

                    if (pClosest == 0) {
                        
                        phiMag = (SOR_Fields(pClosest + 1, 0) * distance - SOR_Fields(pClosest, 0) * (distance - SORa)) / SORa;
                        AMag = (SOR_Fields(pClosest + 1, 1) * distance - SOR_Fields(pClosest, 1) * (distance - SORa)) / SORa;
                    
                    } else if (pClosest < SORnx) {

                        phiMag = (SOR_Fields(pClosest - 1, 0) * (distance - pClosest * SORa) * (distance - (pClosest + 1) * SORa) -
                                  2 * SOR_Fields(pClosest, 0) * (distance - (pClosest - 1) * SORa) * (distance - (pClosest + 1) * SORa) +
                                  SOR_Fields(pClosest + 1, 0) * (distance - (pClosest - 1) * SORa) * (distance - pClosest * SORa)) / (2 * SORa * SORa);
                        AMag = (SOR_Fields(pClosest - 1, 1) * (distance - pClosest * SORa) * (distance - (pClosest + 1) * SORa) -
                                  2 * SOR_Fields(pClosest, 1) * (distance - (pClosest - 1) * SORa) * (distance - (pClosest + 1) * SORa) +
                                  SOR_Fields(pClosest + 1, 1) * (distance - (pClosest - 1) * SORa) * (distance - pClosest * SORa)) / (2 * SORa * SORa);
                    
                    } else {

                        phiMag = 1;
                        if (g == 0) {
                            AMag = 0;
                        } else {
                            AMag = n / g;
                        }
                        // cout << "Off straight string solution grid" << endl;
                    }

                    std::vector<double> nearest_loop_pos = x_string_dt(s, loop_radius, velocity, dt);
                    std::vector<double> s_m(3);
                    std::vector<double> c(3); 
                    std::vector<double> derivative(3);
                    std::vector<double> cross_product(3);

                    s_m[0] = xb - nearest_loop_pos[0];
                    s_m[1] = yb - nearest_loop_pos[1];
                    s_m[2] = zb - nearest_loop_pos[2];

                    // Calculate the magnitude of s_m
                    double s_m_mag = sqrt(pow(s_m[0], 2) + pow(s_m[1], 2) + pow(s_m[2], 2));

                    
                    if (s_m_mag != 0){
                        // Normalize s_m to get its unit vector
                        s_m[0] /= s_m_mag;
                        s_m[1] /= s_m_mag;
                        s_m[2] /= s_m_mag;

                        // Calculate the magnitude of nearest_loop_pos to normalize it
                        double nearest_loop_pos_mag = sqrt(pow(nearest_loop_pos[0], 2) + pow(nearest_loop_pos[1], 2) + pow(nearest_loop_pos[2], 2));

                        // Calculate the unit vector for nearest_loop_pos
                        c[0] = nearest_loop_pos[0] / nearest_loop_pos_mag;
                        c[1] = nearest_loop_pos[1] / nearest_loop_pos_mag;
                        c[2] = nearest_loop_pos[2] / nearest_loop_pos_mag;

                        double dot_product = (c[0] * s_m[0]) + (c[1] * s_m[1]) + (c[2] * s_m[2]);
                        cross_product[0] = (s_m[1] * c[2]) - (s_m[2] * c[1]);
                        cross_product[1] = (s_m[2] * c[0]) - (s_m[0] * c[2]);
                        cross_product[2] = (s_m[0] * c[1]) - (s_m[1] * c[0]);

                        derivative = x_string_dt_derivative(s, loop_radius, velocity, dt);
                        double derivative_mag = sqrt(pow(derivative[0], 2) + pow(derivative[1], 2) + pow(derivative[2], 2));

                        derivative[0] /= derivative_mag;
                        derivative[1] /= derivative_mag;
                        derivative[2] /= derivative_mag;

                        double imag = cross_product[0] * derivative[0] + cross_product[1] * derivative[1] + cross_product[2] * derivative[2];

                        
                        phi[0] = phiMag * dot_product;
                        phi[1] = phiMag * imag;
                    }

                    else{
                        phi[0] = 0;
                        phi[1] = 0;
                    }

                    int zero_value = 0;

                    ic << phi[0] << " " << phi[1] << " " << zero_value << " " << zero_value << " " << zero_value << endl;

                }
            }
        }
    }

    
    else {

        cout << "Unrecognised ic_type" << endl;

    }

    return 0;

}

