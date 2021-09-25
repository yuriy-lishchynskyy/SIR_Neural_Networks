using System;
using System.Collections.Generic;
using System.Text;

namespace SingleNodeModel
{
    class RNumberStore
    {
        public static double SIR(Vector parameters)
        {
            double beta = parameters[0]; // define 2 parameters
            double gamma = parameters[1];

            double r0 = beta / gamma; // calculate R0 value for SIR

            return r0;
        }

        public static double SIIR(Vector parameters)
        {
            double beta = parameters[0]; // define 2 parameters
            double gamma = parameters[1];

            double r0 = beta / (1 * gamma); // calculate R0 value for SIIR

            return r0;
        }

        public static double SIRD(Vector parameters)
        {
            double beta = parameters[0]; // define 3 parameters
            double gamma = parameters[1];
            double mu = parameters[2];

            double r0 = beta / (gamma + mu); // calculate R0 value for SIRD

            return r0;
        }

        public static double SIIRD(Vector parameters)
        {
            double beta = parameters[0]; // define 3 parameters
            double gamma = parameters[1];
            double mu = parameters[2];

            double r0 = beta / ((2 * gamma) + mu); // calculate R0 value for SIIRD

            return r0;
        }

        public static double SEIR(Vector parameters)
        {
            double beta = parameters[0]; // define 3 parameters
            double kappa = parameters[1];
            double gamma = parameters[2];

            double r0 = (beta * kappa) / (gamma * kappa); // calculate R0 value for SEIR

            return r0;
        }

        public static double SEIRD(Vector parameters)
        {
            double beta = parameters[0]; // define 4 parameters
            double kappa = parameters[1];
            double gamma = parameters[2];
            double mu = parameters[3];

            double r0 = (beta * kappa) / ((gamma + mu) * kappa); // calculate R0 value for SEIRD

            return r0;
        }

        public static double SIXRD(Vector parameters)
        {
            double beta = parameters[0]; // define 4 parameters
            double gamma = parameters[1];
            double kappa = parameters[2];
            double mu = parameters[3];

            double r0 = beta / (gamma + kappa + mu); // calculate R0 value for SIXRD

            return r0;
        }
    }
}
