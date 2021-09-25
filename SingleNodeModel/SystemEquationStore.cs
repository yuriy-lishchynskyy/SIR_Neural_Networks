using System;
using System.Collections.Generic;
using System.Text;

namespace SingleNodeModel
{
    class SystemEquationStore
    {
        public static Vector SIR(Vector inputs, Vector parameters, double population) // take in vector of current values of S, I, R and output vector of derivatives
        {
            double beta = parameters[0]; // define 2 parameters
            double gamma = parameters[1];

            double Sj = inputs[0]; // store current values of state equations
            double Ij = inputs[1];
            double Rj = inputs[2];

            Vector dVec = new Vector(3);

            dVec[0] = -(beta * Ij * Sj / population);
            dVec[1] = (beta * Ij * Sj / population) - (gamma * Ij);
            dVec[2] = gamma * Ij;

            return dVec;
        }

        public static Vector SIIR(Vector inputs, Vector parameters, double population) // take in vector of current values of S, I, R and output vector of derivatives
        {
            double beta = parameters[0]; // define 2 parameters
            double gamma = parameters[1];

            double f = parameters[2]; // define proportion of symptomatic / asymptomatic

            double Sj = inputs[0]; // store current values of state equations
            double Isj = inputs[1];
            double Iaj = inputs[2];
            double Rj = inputs[3];

            Vector dVec = new Vector(4); // 4 states

            dVec[0] = -(beta * f * Isj * Sj / population) - (beta * (1 - f) * Iaj * Sj / population); // dS
            dVec[1] = (beta * f * Isj * Sj / population) - (gamma * Isj); // dIs
            dVec[2] = (beta * (1 - f) * Iaj * Sj / population) - (gamma * Iaj); // dIa
            dVec[3] = (gamma * Isj) + (gamma * Iaj); // dR

            return dVec;
        }

        public static Vector SIRD(Vector inputs, Vector parameters, double population)
        {
            double beta = parameters[0]; // define 3 parameters
            double gamma = parameters[1];
            double mu = parameters[2];

            double Sj = inputs[0]; // store current values of state equations
            double Ij = inputs[1];
            double Rj = inputs[2];
            double Dj = inputs[3];

            Vector dVec = new Vector(4);

            dVec[0] = -(beta * Ij * Sj / population);
            dVec[1] = (beta * Ij * Sj / population) - (gamma * Ij) - (mu * Ij);
            dVec[2] = gamma * Ij;
            dVec[3] = mu * Ij;

            return dVec;
        }

        public static Vector SIIRD(Vector inputs, Vector parameters, double population)
        {
            double beta = parameters[0]; // define 3 parameters
            double gamma = parameters[1];
            double mu = parameters[2];

            double f = parameters[3]; // define proportion of symptomatic / asymptomatic

            double Sj = inputs[0]; // store current values of state equations
            double Isj = inputs[1];
            double Iaj = inputs[2];
            double Rj = inputs[3];
            double Dj = inputs[4];

            Vector dVec = new Vector(4); // 4 states

            dVec[0] = -(beta * f * Isj * Sj / population) - (beta * (1 - f) * Iaj * Sj / population); // dS
            dVec[1] = (beta * f * Isj * Sj / population) - (gamma * Isj) - (mu * Isj); // dIs
            dVec[2] = (beta * (1 - f) * Iaj * Sj / population) - (gamma * Iaj); // dIa
            dVec[3] = (gamma * Isj) + (gamma * Iaj); // dR
            dVec[4] = (mu * Isj); // dD

            return dVec;
        }

        public static Vector SEIR(Vector inputs, Vector parameters, double population)
        {
            double beta = parameters[0]; // define 3 parameters
            double lambda = parameters[1];
            double gamma = parameters[2];

            double Sj = inputs[0]; // store current values of state equations
            double Ej = inputs[1];
            double Ij = inputs[2];
            double Rj = inputs[3];

            Vector dVec = new Vector(4);

            dVec[0] = -(beta * Ij * Sj / population);
            dVec[1] = (beta * Ij * Sj / population) - (lambda * Ej);
            dVec[2] = (lambda * Ej) - (gamma * Ij);
            dVec[3] = gamma * Ij;

            return dVec;
        }

        public static Vector SEIRD(Vector inputs, Vector parameters, double population)
        {
            double beta = parameters[0]; // define 4 parameters
            double lambda = parameters[1];
            double gamma = parameters[2];
            double mu = parameters[3];

            double Sj = inputs[0]; // store current values of state equations
            double Ej = inputs[1];
            double Ij = inputs[2];
            double Rj = inputs[3];
            double Dj = inputs[4];

            Vector dVec = new Vector(5);

            dVec[0] = -(beta * Ij * Sj / population);
            dVec[1] = (beta * Ij * Sj / population) - (lambda * Ej);
            dVec[2] = (lambda * Ej) - (gamma * Ij) - (mu * Ij);
            dVec[3] = gamma * Ij;
            dVec[4] = mu * Ij;

            return dVec;
        }

        public static Vector SIXRD(Vector inputs, Vector parameters, double population)
        {
            double beta = parameters[0]; // define 4 parameters
            double gamma = parameters[1];
            double kappa = parameters[2];
            double mu = parameters[3];

            double Sj = inputs[0]; // store current values of state equations
            double Ij = inputs[1];
            double Xj = inputs[2];
            double Rj = inputs[3];
            double Dj = inputs[4];

            Vector dVec = new Vector(5);

            dVec[0] = -(beta * Ij * Sj / population);
            dVec[1] = (beta * Ij * Sj / population) - (kappa * Ij) - (gamma * Ij) - (mu * Ij);
            dVec[2] = (kappa * Ij) - (gamma * Xj) - (mu * Xj);
            dVec[3] = (gamma * Ij) + (gamma * Xj);
            dVec[4] = (mu * Ij) + (mu * Xj);

            return dVec;
        }
    }
}
