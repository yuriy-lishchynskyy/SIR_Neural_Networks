using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace SingleNodeModel
{
    public class RungeKutta
    {
        // data
        private int n_steps;
        private double stepsize = 0.1; // for runge-kutta
        private double writesize = 0.1; // for writing to csv
        private Vector initial_values;
        private double t_final = 100;
        private int calc_e = 0;

        private double population = 4500000; // total population
        private double population0 = 4500000; // initial total population

        private double[] xvals; // store x = t
        private Vector[] yvals; // store y's
        private double[] rvals; // store R0 number

        public double Stepsize
        {
            get
            {
                return stepsize;
            }
            set
            {
                if (value > 0 & value <= t_final)
                {
                    stepsize = value;
                    n_steps = Convert.ToInt32(t_final / stepsize);
                    this.xvals = new double[n_steps + 1];
                    this.yvals = new Vector[n_steps + 1];
                    this.rvals = new double[n_steps + 1];
                }
                else
                {
                    string e = String.Format("ERROR: Invalid Step Size selected ({0}) - must be positive and less than Final Time ({1})", value, t_final);
                    throw new Exception(e);
                }
            }
        }

        public double Writesize
        {
            get
            {
                return writesize;
            }
            set
            {
                if (value > 0 & value <= t_final)
                {
                    writesize = value;
                }
                else
                {
                    string e = String.Format("ERROR: Invalid Write Size selected ({0}) - must be positive and less than Final Time ({1})", value, t_final);
                    throw new Exception(e);
                }
            }
        }

        public double T_final
        {
            get
            {
                return t_final;
            }
            set
            {
                if (value > 0 & value > stepsize)
                {
                    t_final = value;
                    n_steps = Convert.ToInt32(t_final / stepsize);
                    this.xvals = new double[n_steps + 1];
                    this.yvals = new Vector[n_steps + 1];
                    this.rvals = new double[n_steps + 1];
                }
                else
                {
                    string e = String.Format("ERROR: Invalid Final Time selected ({0}) - must be positive and greater than Step Size ({1})", value, stepsize);
                    throw new Exception(e);
                }
            }
        }

        // constructors
        public RungeKutta()
        {

        }

        public RungeKutta(Vector values0) // initialise object with vector of starting values (length of which infers no. system equations)
        {
            this.n_steps = Convert.ToInt32(this.t_final / this.stepsize);

            this.xvals = new double[n_steps + 1];
            this.yvals = new Vector[n_steps + 1];
            this.rvals = new double[n_steps + 1];

            this.initial_values = values0;

            this.population = values0.Sum();

            this.population0 = this.population;
        }

        // methods
        public void MonteCarlo(ODESystem system, RSystem rnumber, string type, string location_output, int m_generate, double r_min, double r_max) // generate parameters + I data
        {
            int calc_e = 0;
            int p = type.Length - 1; // number of parameters (beta/gamma/mu etc)

            Random gen_beta = new Random();
            Random gen_gamma = new Random();
            Random gen_n = new Random();

            double min_beta = 0.0; // limits on min/max for beta
            double max_beta = 1.0;

            double min_gamma = 1.0 / 15; // limits on min/max for gamma
            double max_gamma = 1.0 / 5;

            double min_n = 0.1 / 100; // limits on min/max for population ratio
            double max_n = 5.0 / 100;

            double[] parameters = new double[p];

            StreamWriter sw_gen = null;

            int n_data = xvals.Length; // how much data to write

            int indx_i = 0;
            int indx_d = 0;

            double r0 = 3; // R0 number

            if (type == "SIR" || type == "SIIR" || type ==  "SIRD" || type == "SIXRD") // index of Infected data in "yvals" vector
            {
                indx_i = 1;
                
                if (type == "SIRD")
                {
                    indx_d = 3;
                }
                else if (type == "SIIRD")
                {
                    indx_d = 4;
                }
                else if (type == "SIXRD")
                {
                    indx_d = 4;
                }    
            }
            else if (type == "SEIR" || type == "SEIRD")
            {
                indx_i = 2;
            }

            double n_ratio = 0;
            double beta = 0;
            double gamma = 0;

            try // enclose problematic code in try block to throw exception if any part fails
            {
                sw_gen = new StreamWriter(location_output);

                for (int i = 0; i < m_generate; i++) // CREATE DATASET
                {
                    string content = ""; // string to store content to be written to csv row

                    if (true) // modify starting population
                    {
                        n_ratio = min_n + (gen_n.NextDouble() * (max_n - min_n));

                        this.population = n_ratio * this.population0;
                        this.initial_values = new Vector(new double[] { this.population - this.initial_values[1], this.initial_values[1], 0.00 });                       
                    }

                    do
                    {
                        //beta = min_beta + (gen_beta.NextDouble() * (max_beta - min_beta));
                        beta = gen_beta.NextDouble();

                        //gamma = min_gamma + (gen_gamma.NextDouble() * (max_gamma - min_gamma));
                        gamma = 1.0 / 11;
                        // gamma = gen_gamma.NextDouble();

                        Vector parameters2 = new Vector(new double[] { beta, gamma });
                        r0 = rnumber(parameters2);
                    }
                    while (r0 < r_min || r0 > r_max); // ensure parameters result in realistic R0

                    parameters[0] = beta;
                    parameters[1] = gamma;

                    this.RK1(system, rnumber, parameters); // run solver

                    content += "666,"; // IDENTIFIER - parameters
                    
                    if (true)
                    {
                        content += n_ratio.ToString() + ",";
                    }
                    
                    for (int j = 0; j < p; j++) // add parameters to csv row
                    {
                        content += parameters[j].ToString() + ",";
                    }

                    content += "777,"; // IDENTIFIER - infected

                    for (int j = 0; j < n_data; j++) // I-values to csv row
                    {
                        content += yvals[j][indx_i] + ",";
                    }

                    if (type == "SIRD" || type == "SIIRD" || type == "SIXRD")
                    {
                        content += "888,"; // IDENTIFIER - dead

                        for (int j = 0; j < n_data; j++) // I-values to csv row
                        {
                            content += yvals[j][indx_d] + ",";
                        }
                    }

                    content += "999"; // IDENTIFIER - end

                    sw_gen.WriteLine(content); // write data row to csv

                    Console.WriteLine("Iteration {0} / {1}", (i + 1) / (this.writesize / this.stepsize), m_generate);
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("ERROR: {0}", e.Message); // print error message
                calc_e = 1;
            }
            finally
            {
                if (sw_gen != null) // only close file if no errors after writing file
                {
                    sw_gen.Close();

                    Console.WriteLine("-------------------------------");

                    if (calc_e != 1)
                    {
                        Console.WriteLine("DATA EXPORT TO CSV: SUCCESSFUL");
                    }
                    else
                    {
                        Console.WriteLine("DATA EXPORT TO CSV: UNSUCCESSFUL");
                    }
                }
            }
        }

        public void RK1(ODESystem system, RSystem rnumber, double[,] parameters_table) // table of parameters (varying)
        {
            calc_e = 0;

            try
            {
                int m = this.initial_values.Length; // size of system (no. of equations)

                int p = parameters_table.GetLength(0); // number of parameters rad from csv
                Vector parameters = new Vector(p);

                for (int j = 0; j < p; j++)
                {
                    parameters[j] = parameters_table[j, 0];
                }

                double xj = 0; // value of current t (starting value = 0)
                double xj1; // value of next t

                Vector yj = this.initial_values; // vector of current state of system (starting value = input)
                Vector yj1 = new Vector(m); // vector of next state of system

                double r0j = rnumber(parameters); // R0 value

                xvals[0] = xj;
                yvals[0] = yj;
                rvals[0] = r0j;

                Vector k1 = new Vector(m); // vector of k1/2/3/4 values for each system variable

                for (int i = 1; i < n_steps + 1; i++)
                {
                    for (int j = 0; j < p; j++) // create vector of parameters for current timestep
                    {
                        parameters[j] = parameters_table[j, i - 1];
                    }

                    k1 = system(yj, parameters, this.population);

                    xj1 = xj + this.stepsize;
                    yj1 = yj + (this.stepsize / 1) * (k1);

                    r0j = rnumber(parameters);

                    xvals[i] = xj1;
                    yvals[i] = yj1;
                    rvals[i] = r0j;

                    xj = xj1; // set "next" state of system as "current" state for next loop
                    yj = yj1;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("ERROR: {0}", e.Message); // print error message
                calc_e = 1;
            }
        }

        public void RK1(ODESystem system, RSystem rnumber, double[] parameters_table) // vector of parameters (fixed)
        {
            calc_e = 0;

            try
            {
                int m = this.initial_values.Length; // size of system (no. of equations)

                int p = parameters_table.GetLength(0); // number of parameters rad from csv
                Vector parameters = new Vector(parameters_table);

                double xj = 0; // value of current t (starting value = 0)
                double xj1; // value of next t

                Vector yj = this.initial_values; // vector of current state of system (starting value = input)
                Vector yj1 = new Vector(m); // vector of next state of system

                double r0j = rnumber(parameters); // R0 value

                xvals[0] = xj;
                yvals[0] = yj;
                rvals[0] = r0j;

                Vector k1 = new Vector(m); // vector of k1/2/3/4 values for each system variable

                for (int i = 1; i < n_steps + 1; i++)
                {
                    k1 = system(yj, parameters, this.population);

                    xj1 = xj + this.stepsize;
                    yj1 = yj + (this.stepsize / 1) * (k1);

                    r0j = rnumber(parameters);

                    xvals[i] = xj1;
                    yvals[i] = yj1;
                    rvals[i] = r0j;

                    xj = xj1; // set "next" state of system as "current" state for next loop
                    yj = yj1;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("ERROR: {0}", e.Message); // print error message
                calc_e = 1;
            }
        }

        public void RK4(ODESystem system, RSystem rnumber, double[,] parameters_table) // table of parameters (varying)
        {
            calc_e = 0;

            try
            {
                int m = this.initial_values.Length; // size of system (no. of equations)

                int p = parameters_table.GetLength(0); // number of parameters rad from csv
                Vector parameters = new Vector(p);

                for (int j = 0; j < p; j++)
                {
                    parameters[j] = parameters_table[j, 0];
                }

                double xj = 0; // value of current t (starting value = 0)
                double xj1; // value of next t

                Vector yj = this.initial_values; // vector of current state of system (starting value = input)
                Vector yj1 = new Vector(m); // vector of next state of system

                double r0j = rnumber(parameters); // R0 value

                xvals[0] = xj;
                yvals[0] = yj;
                rvals[0] = r0j;

                Vector k1 = new Vector(m); // vector of k1/2/3/4 values for each system variable
                Vector k2 = new Vector(m);
                Vector k3 = new Vector(m);
                Vector k4 = new Vector(m);

                for (int i = 1; i < n_steps + 1; i++)
                {
                    for (int j = 0; j < p; j++) // create vector of parameters for current timestep
                    {
                        parameters[j] = parameters_table[j, i];
                    }

                    k1 = system(yj, parameters, this.population);
                    k2 = system(yj + ((this.stepsize / 2) * k1), parameters, this.population);
                    k3 = system(yj + ((this.stepsize / 2) * k2), parameters, this.population);
                    k4 = system(yj + (this.stepsize * k3), parameters, this.population);

                    xj1 = xj + this.stepsize;
                    yj1 = yj + (this.stepsize / 6) * (k1 + (2 * k2) + (2 * k3) + k4);

                    r0j = rnumber(parameters);

                    xvals[i] = xj1;
                    yvals[i] = yj1;
                    rvals[i] = r0j;

                    xj = xj1; // set "next" state of system as "current" state for next loop
                    yj = yj1;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("ERROR: {0}", e.Message); // print error message
                calc_e = 1;
            }
        }

        public void RK4(ODESystem system, RSystem rnumber, double[] parameters_table) // vector of parameters (fixed)
        {
            calc_e = 0;

            try
            {
                int m = this.initial_values.Length; // size of system (no. of equations)

                int p = parameters_table.GetLength(0); // number of parameters rad from csv
                Vector parameters = new Vector(parameters_table);

                double xj = 0; // value of current t (starting value = 0)
                double xj1; // value of next t

                Vector yj = initial_values; // vector of current state of system (starting value = input)
                Vector yj1 = new Vector(m); // vector of next state of system

                double r0j = rnumber(parameters); // R0 value

                xvals[0] = xj;
                yvals[0] = yj;
                rvals[0] = r0j;

                Vector k1 = new Vector(m); // vector of k1/2/3/4 values for each system variable
                Vector k2 = new Vector(m);
                Vector k3 = new Vector(m);
                Vector k4 = new Vector(m);

                for (int i = 1; i < n_steps + 1; i++)
                {
                    k1 = system(yj, parameters, this.population);
                    k2 = system(yj + ((this.stepsize / 2) * k1), parameters, this.population);
                    k3 = system(yj + ((this.stepsize / 2) * k2), parameters, this.population);
                    k4 = system(yj + (this.stepsize * k3), parameters, this.population);

                    xj1 = xj + this.stepsize;
                    yj1 = yj + (this.stepsize / 6) * (k1 + (2 * k2) + (2 * k3) + k4);

                    r0j = rnumber(parameters);

                    xvals[i] = xj1;
                    yvals[i] = yj1;
                    rvals[i] = r0j;

                    xj = xj1; // set "next" state of system as "current" state for next loop
                    yj = yj1;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("ERROR: {0}", e.Message); // print error message
                calc_e = 1;
            }
        }

        public void WriteCSV(string location, string type, double[,] parameters)
        {
            StreamWriter sw = null;
            int n_data = Convert.ToInt32(this.t_final / writesize); // how much data to write

            try // enclose problematic code in try block to throw exception if any part fails
            {
                sw = new StreamWriter(location);

                char[] states = type.ToCharArray(); // convert model type to individual state letters

                string content = "t,"; // string to store content to be written to csv row

                for (int j = 0; j < yvals[0].Length; j++) // create header row (t and m no. y's - depending on size of system)
                {
                    content += states[j] + ",";
                }

                content += "R0";

                if (type == "SIR")
                {
                    content += ",beta, gamma";
                }
                if (type == "SIIR")
                {
                    content += ",beta, gamma, f";
                }
                else if (type == "SIRD")
                {
                    content += ",beta, gamma, mu";
                }
                if (type == "SIIRD")
                {
                    content += ",beta, gamma, mu, f";
                }
                else if (type == "SIXRD")
                {
                    content += ",beta, kappa, gamma, mu";
                }
                else if (type == "SEIR")
                {
                    content += ",beta, lambda, gamma";
                }
                else if (type == "SEIRD")
                {
                    content += ",beta, lambda, gamma, mu";
                }

                sw.WriteLine(content); // write header to csv

                for (int i = 0; i < n_data; i++)
                {
                    if (i % (this.writesize / this.stepsize) == 0)
                    {
                        Vector ydata = yvals[i];

                        content = "";

                        content = xvals[i].ToString() + ","; // add x-value (t)

                        for (int j = 0; j < ydata.Length; j++)
                        {
                            content += ydata[j].ToString() + ",";
                        }

                        content += rvals[i].ToString();

                        for (int j = 0; j < parameters.GetLength(0); j++) // add parameter values
                        {
                            content += "," + parameters[j, i];
                        }

                        sw.WriteLine(content); // write data row to csv
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("ERROR: {0}", e.Message); // print error message
            }
            finally
            {
                if (sw != null) // only close file if no errors after writing file
                {
                    sw.Close();
                    Console.WriteLine("-------------------------------");

                    if (calc_e != 1)
                    {
                        Console.WriteLine("DATA EXPORT TO CSV: SUCCESSFUL");
                    }
                    else
                    {
                        Console.WriteLine("DATA EXPORT TO CSV: UNSUCCESSFUL");
                    }

                    Console.WriteLine();
                }
            }
        }
    }
}
