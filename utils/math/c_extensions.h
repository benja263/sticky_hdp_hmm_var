//
// Created by Benjamin Fuhrer on 26/06/2020.
//

#ifndef EXTENDEDMATH_C_EXTENSIONS_H
#define EXTENDEDMATH_C_EXTENSIONS_H

#include <cstddef>
#include <math.h>

extern "C"{
double ExtendedLog(double x);
double ExtendedExp(double x);
double ExtendedLogSum(double log_x, double log_y);
double ExtendedLogProduct(double log_x, double log_y);
void ExtendedArrayLog(double** x, unsigned int rows, unsigned int cols);
void ExtendedArrayExp(double** x, unsigned int rows, unsigned int cols);

void forwards(double** log_alpha, double** obs_likelihood, double** log_pi_z,
                 unsigned int rows, unsigned int cols);

void backwards(double** log_beta,  double** obs_likelihood, double** log_pi_z,
                  unsigned int rows, unsigned int cols);


void rand_gamma(double* output, double* param, unsigned int shape);

void rand_dirichlet(double* output, double* alpha, unsigned int shape);
};
#endif //EXTENDEDMATH_C_EXTENSIONS_H