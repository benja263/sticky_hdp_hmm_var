//
// Created by Benjamin Fuhrer on 26/06/2020.
//

#include "c_extensions.h"
#include "random"

    double ExtendedLog(double x) {
        if (x == 0.0) {
            return nan("");
        }
        return log(x);
    }

    double ExtendedExp(double x) {
        if (isnan(x)) {
            return 0.0;
        }
        return exp(x);
    }

    double ExtendedLogSum(double log_x, double log_y) {
        if (isnan(log_x) || isnan(log_y)) {
            if (isnan(log_x)) {
                return log_y;
            }
            return log_x;
        }
        if (log_x > log_y) {
            return log_x + ExtendedLog(1 + exp(log_y - log_x));
        }
        return log_y + ExtendedLog(1 + exp(log_x - log_y));
    }

    double ExtendedLogProduct(double log_x, double log_y) {
        if (isnan(log_x) || isnan(log_y)) {
            return nan("");
        }
        return log_x + log_y;
    }


    void ExtendedArrayLog(double** x, unsigned int rows, unsigned int cols)
    {
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                x[i][j] = ExtendedLog(x[i][j]);
            }
        }
    }

    void ExtendedArrayExp(double** x, unsigned int rows, unsigned int cols)
    {
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                x[i][j] = ExtendedExp(x[i][j]);
            }
        }
    }


    void forwards(double** log_alpha, double** obs_likelihood, double** log_pi_z, unsigned int rows,
                            unsigned int cols) {
        for (unsigned int t = 0; t < cols - 1; t++) {
            for (unsigned int j = 0; j < rows; j++) {
                double accumulator = nan("");
                for (unsigned int i = 0; i < rows; i++) {
                    accumulator = ExtendedLogSum(accumulator,
                                                               ExtendedLogProduct(log_alpha[i][t],
                                                                                                log_pi_z[i][j]));
                }
                log_alpha[j][t + 1] = ExtendedLogProduct(accumulator, obs_likelihood[j][t]);
            }
        }
    }

    void backwards(double** log_beta, double** obs_likelihood, double** log_pi_z,
                             unsigned int rows, unsigned int cols) {
        for (int t = cols - 2; t >= 0; t--) {
            for (unsigned int i = 0; i < rows; i++) {
                double accumulator = nan("");
                for (unsigned int j = 0; j < rows; j++) {
                    accumulator = ExtendedLogSum(accumulator, ExtendedLogProduct(log_pi_z[i][j],
                            ExtendedLogProduct(obs_likelihood[j][t+1], log_beta[j][t+1])));
                }
                log_beta[i][t] = accumulator;
            }
        }
    }

    void rand_gamma(double* output, double* param, unsigned int shape)
    {

        std::default_random_engine generator;
        double a = 1.0;

        for (unsigned int i = 0; i < shape; i++)
        {
            std::gamma_distribution<double> distribution(a, param[i]);
            output[i] = distribution(generator);
        }
    }

    void rand_dirichlet(double* output, double* alpha, unsigned int shape)
    {
        rand_gamma(output, alpha, shape);
        double sum = 0.0;
        for (unsigned int i = 0; i < shape; i++)
        {
            sum += output[i];
        }
        if (sum == 0.0)
        {
            for (unsigned int i = 0; i < shape; i++)
            {
                alpha[i] += 1.0;
            }
            rand_gamma(output, alpha, shape);
            for (unsigned int i = 0; i < shape; i++) {
                sum += output[i];
            }
        }
        for (unsigned int i = 0; i < shape; i++)
        {
            output[i] /= sum;
        }
    }

