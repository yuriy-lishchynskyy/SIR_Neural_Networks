using System;
using System.Collections.Generic;
using System.Text;

namespace SingleNodeModel
{
    public class Vector
    {
        // data
        private double[] data;
        private int length;

        public int Length { get => this.length; }

        // constructors
        public Vector()
        {

        }

        public Vector(int l) // create zero vector of length l
        {
            if (l > 0)
            {
                this.length = l;
                this.data = new double[this.length];

                for (int i = 0; i < this.length; i++)
                {
                    this.data[i] = 0;
                }
            }
            else
            {
                Console.WriteLine("Invalid input for Vector length");
            }
        }

        public Vector(double[] a) // create vector from an array
        {
            this.length = a.GetLength(0);
            this.data = new double[this.length];

            for (int i = 0; i < this.length; i++)
            {
                this.data[i] = a[i];
            }
        }

        public Vector(Vector a) // create vector from another vector
        {
            this.length = a.Length;
            this.data = new double[this.length];

            for (int i = 0; i < this.length; i++)
            {
                this.data[i] = a[i];
            }
        }

        // overloaded operators
        public double this[int i] // indexing operator
        {
            get
            {
                return this.data[i];
            }
            set
            {
                this.data[i] = value;
            }
        }

        public static Vector operator +(Vector a, Vector b) // addition of 2 vectors
        {
            int i;

            int a_l = a.Length;
            int b_l = b.Length;

            if (a_l != b_l) // check if vectors same size
            {
                Console.WriteLine("Cannot add vectors: Lengths not equal!");
                return null;
            }

            Vector temp = new Vector(a_l);

            for (i = 0; i < a_l; i++) // add corresponding elements
            {
                temp[i] = a[i] + b[i];
            }

            return temp;
        }

        public static Vector operator -(Vector a, Vector b) // subtraction of 2 vectors
        {
            int i;

            int a_l = a.Length;
            int b_l = b.Length;

            if (a_l != b_l) // check if vectors same size
            {
                Console.WriteLine("Cannot subtract vectors: Lengths not equal!");
                return null;
            }

            Vector temp = new Vector(a_l);

            for (i = 0; i < a_l; i++) // subtract corresponding elements
            {
                temp[i] = a[i] - b[i];
            }

            return temp;
        }

        public static double operator *(Vector a, Vector b) // dot product multiplication of 2 vectors
        {
            int i;

            int a_l = a.Length;
            int b_l = b.Length;

            if (a_l != b_l) // check if vectors same size
            {
                Console.WriteLine("Cannot dot multiply vectors: Lengths not equal!");
                return 0;
            }

            double sum = 0;

            for (i = 0; i < a_l; i++) // running sum of multiplied terms
            {
                sum = sum + (a[i] * b[i]);
            }

            return sum;
        }

        public static Vector operator *(Vector a, double x) // multiplication of vector by constant
        {
            int i;
            int a_l = a.Length;

            Vector temp = new Vector(a_l);

            for (i = 0; i < a_l; i++) // scale each term by constant
            {
                temp[i] = a[i] * x;
            }

            return temp;
        }

        public static Vector operator *(double x, Vector a) // multiplication of constant by vector
        {
            int i;
            int a_l = a.Length;

            Vector temp = new Vector(a_l);

            for (i = 0; i < a_l; i++) // scale each term by constant
            {
                temp[i] = a[i] * x;
            }

            return temp;
        }

        // methods
        public double Sum()
        {
            double sum = 0;

            for (int i = 0; i < this.length; i++)
            {
                sum += this.data[i];
            }

            return sum;
        }
    }
}
