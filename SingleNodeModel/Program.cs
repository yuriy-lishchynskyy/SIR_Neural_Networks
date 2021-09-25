using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace SingleNodeModel
{
    public delegate Vector ODESystem(Vector inputs, Vector parameters, double population); // delegate with signature for passing systems of equations into Runge-Kutta class

    public delegate double RSystem(Vector parameters); // delegate with signature for passing system formulas for R0 number into Runge-Kutta class

    class Program
    {
        static void Main(string[] args)
        {
            // ---------------------------------------------------------------------------------
            // ---------------------------------------------------------------------------------
            // INITIALISATION
            double pop;
            double i0;

            Vector init_values_sir;
            Vector init_values_sird;

            // ---------------------------------------------------------------------------------
            // ---------------------------------------------------------------------------------
            // PINN
            pop = 100;
            i0 = 1;

            init_values_sir = new Vector(new double[] { pop - i0, i0, 0.00 });
            init_values_sird = new Vector(new double[] { pop - i0, i0, 0.00, 0.00 });

            // ---------------------------------------------------------------------------------
            // PINN - FIXED
            string parameters_sir_fixed = "SingleNodeModel input//parameters_sir_fixed.csv";
            string parameters_sird_fixed = "SingleNodeModel input//parameters_sird_fixed.csv";

            // PINN - FIXED OUTPUT
            if (true)
            {
                Console.WriteLine("SIR");
                SimulateSIR(init_values_sir, parameters_sir_fixed);

                Console.WriteLine("SIRD");
                SimulateSIRD(init_values_sird, parameters_sird_fixed);
            }
            
            // ---------------------------------------------------------------------------------
            // PINN - VARIED
            string[] parameter_sir_varied_small = new string[] { "SingleNodeModel input//parameters_sir_beta_constant_small.csv",
                                                                 "SingleNodeModel input//parameters_sir_beta_step1_small.csv",
                                                                 "SingleNodeModel input//parameters_sir_beta_step2_small.csv",
                                                                 "SingleNodeModel input//parameters_sir_beta_step1_gamma_step1_small.csv",
                                                                 "SingleNodeModel input//parameters_sir_beta_expo_small.csv" };

            string[] parameter_sir_varied_large = new string[] { "SingleNodeModel input//parameters_sir_beta_constant_large.csv",
                                                                 "SingleNodeModel input//parameters_sir_beta_step1_large.csv",
                                                                 "SingleNodeModel input//parameters_sir_beta_step2_large.csv",
                                                                 "SingleNodeModel input//parameters_sir_beta_step1_gamma_step1_large.csv",
                                                                 "SingleNodeModel input//parameters_sir_beta_expo_large.csv" };

            string[] parameter_sird_varied_small = new string[] { "SingleNodeModel input//parameters_sird_beta_constant_small.csv",
                                                                  "SingleNodeModel input//parameters_sird_beta_step1_small.csv",
                                                                  "SingleNodeModel input//parameters_sird_beta_step2_small.csv",
                                                                  "SingleNodeModel input//parameters_sird_beta_step1_gamma_step1_mu_step1_small.csv",
                                                                  "SingleNodeModel input//parameters_sird_beta_expo_small.csv" };

            // PINN - VARIED OUTPUT
            if (true)
            {
                for (int i = 0; i < parameter_sir_varied_small.Length; i++)
                {
                    Console.WriteLine("SIR Small - {0}", i + 1);
                    SimulateSIR(init_values_sir, parameter_sir_varied_small[i]);
                }

                for (int i = 0; i < parameter_sir_varied_large.Length; i++)
                {
                    Console.WriteLine("SIR Large - {0}", i + 1);
                    SimulateSIR(init_values_sir, parameter_sir_varied_large[i]);
                }

                for (int i = 0; i < parameter_sird_varied_small.Length; i++)
                {
                    Console.WriteLine("SIRD Small - {0}", i + 1);
                    SimulateSIRD(init_values_sird, parameter_sird_varied_small[i]);
                }
            }

            // ---------------------------------------------------------------------------------
            // ---------------------------------------------------------------------------------
            // MONTE CARLO
            int n_train = 6250;
            int n_test = 1;

            int r_min = 1;
            int r_max = 5;

            pop = 4500000;
            i0 = 1;

            init_values_sir = new Vector(new double[] { pop - i0, i0, 0.00 });

            // ---------------------------------------------------------------------------------
            // MONTE CARLO - OVERALL
            string filename_sir_train = "SingleNodeModel output//output_sir_monte_carlo_train.csv";
            string filename_sir_test = "SingleNodeModel output//output_sir_monte_carlo_test.csv";

            // MONTE CARLO - OVERALL OUTPUT
            if (true)
            {
                Console.WriteLine("GENERATING MONTE CARLO DATASET - SIR");
                
                SimulateSIR_MC(init_values_sir, n_train, r_min, r_max, filename_sir_train);
                SimulateSIR_MC(init_values_sir, n_test, r_min, r_max, filename_sir_test);
            }          

        }
        public static void SimulateSIR(Vector init_values_sir, string parameters_sir_fixed)
        {
            RungeKutta rk_sir = new RungeKutta(init_values_sir); // Initialize RungeKutta solver class
            rk_sir.Stepsize = 1;
            rk_sir.Writesize = 1;
            rk_sir.T_final = 500;

            double[,] param_sir_fixed = ReadData(parameters_sir_fixed, "SIR");

            rk_sir.RK1(SystemEquationStore.SIR, RNumberStore.SIR, param_sir_fixed);

            int len_input = parameters_sir_fixed.Split("//").Length; // determine output string based on input (parameter) string
            string input_name = parameters_sir_fixed.Split("//")[len_input - 1];

            string out_name = "output" + input_name.Substring(10, input_name.Length - 10);
            string output_sir_fixed = "SingleNodeModel output//" + out_name;

            rk_sir.WriteCSV(output_sir_fixed, "SIR", param_sir_fixed);
            Console.WriteLine("");
        }

        public static void SimulateSIRD(Vector init_values_sird, string parameters_sird_fixed)
        {
            RungeKutta rk_sird = new RungeKutta(init_values_sird); // Initialize RungeKutta solver class
            rk_sird.Stepsize = 1;
            rk_sird.Writesize = 1;
            rk_sird.T_final = 500;

            double[,] param_sird_fixed = ReadData(parameters_sird_fixed, "SIRD");

            rk_sird.RK1(SystemEquationStore.SIRD, RNumberStore.SIRD, param_sird_fixed);

            int len_input = parameters_sird_fixed.Split("//").Length; // determine output string based on input (parameter) string
            string input_name = parameters_sird_fixed.Split("//")[len_input - 1];

            string out_name = "output" + input_name.Substring(10, input_name.Length - 10);
            string output_sird_fixed = "SingleNodeModel output//" + out_name;

            rk_sird.WriteCSV(output_sird_fixed, "SIRD", param_sird_fixed);
            Console.WriteLine("");
        }

        public static void SimulateSIR_MC(Vector init_values_sir, int n_train, int r_min, int r_max, string filename_sir)
        {
            RungeKutta rkm_sir = new RungeKutta(init_values_sir); // Initialize RungeKutta solver class
            rkm_sir.Stepsize = 1;
            rkm_sir.Writesize = 1;
            rkm_sir.T_final = 365;

            rkm_sir.MonteCarlo(SystemEquationStore.SIR, RNumberStore.SIR, "SIR", filename_sir, n_train, r_min, r_max);
            Console.WriteLine("");
        }

        public static double[,] ReadData(string filename, string type) // read parameter values from csv
        {
            StreamReader sr = null;
            string temp = null; // used to read each line
            char[] char_separators = new char[] { ',' };
            string[] output = null; // store each line by breaking each row of 4 items into 4 pieces and storing each piece in array

            double a;
            int p = type.Length - 1; // length of "type" string = number of parameters to expect in csv - 

            double[,] param_store = new double[1,1]; // p no. of parameters

            try // enclose problematic code in try block to throw exception if any part fails
            {
                sr = new StreamReader(filename);

                var lines = File.ReadAllLines(filename);
                int len = lines.Length - 1; // no. of values in timeseries to expect

                int i = 0; // row counter                

                param_store = new double[p, len];

                do
                {
                    temp = sr.ReadLine();

                    if (temp == null)
                    {
                        break; // stop if line is empty (ie end of csv)
                    }

                    output = temp.Split(char_separators); // separates "temp" string by each item in array (in this case - comma, comma, comma etc. If 2 items in array, would alternate)

                    bool success = double.TryParse(output[1], out a); // use to ignore text-based header row

                    if (success)
                    {
                        for (int j = 0; j < p; j++)
                        {
                            param_store[j, i] = Convert.ToDouble(output[j]);
                        }

                        i++;
                    }

                } while (true);
            }
            catch (Exception e)
            {
                Console.WriteLine("Error {0}", e.Message); // print error messag
            }
            finally
            {
                if (sr != null) // only close file if no errors after reading file
                    sr.Close();
            }

            return param_store;
        }
    }
}
