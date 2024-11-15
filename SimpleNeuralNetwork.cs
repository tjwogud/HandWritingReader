using System.Collections.Generic;

namespace HandWritingReader
{
    public class SimpleNeuralNetwork
    {
        public struct TrainingData(double[] input, double[] output)
        {
            public double[] input = input;
            public double[] output = output;
        }

        private int[] sizes;
        private Matrix[] biases;
        private Matrix[] weights;

        public SimpleNeuralNetwork(string path)
        {
            using FileStream fs = new FileStream(path, FileMode.Open);
            using BinaryReader bw = new BinaryReader(fs);
            sizes = new int[bw.ReadInt32()];
            for (int i = 0; i < sizes.Length; i++)
                sizes[i] = bw.ReadInt32();
            biases = new Matrix[sizes.Length - 1];
            for (int i = 0; i < sizes.Length - 1; i++)
                biases[i] = new Matrix(sizes[i + 1], 1, (m, n) => Simplify(bw.ReadDouble()));
            weights = new Matrix[sizes.Length - 1];
            for (int i = 0; i < sizes.Length - 1; i++)
                weights[i] = new Matrix(sizes[i + 1], sizes[i], (m, n) => Simplify(bw.ReadDouble()));
        }

        public double[] FeedForward(double[] input)
        {
            Matrix a = Matrix.FromArray(input, 1);
            for (int i = 0; i < a.M * a.N; i++)
                a[i] = Simplify(a[i]);
            for (int i = 0; i < sizes.Length - 1; i++)
            {
                a = Activation(weights[i].Dot(a) + biases[i]);
            }
            return (double[])a;
        }

        private static Matrix Activation(Matrix x)
        {
            Matrix result = new Matrix(x.M, x.N);
            for (int i = 0; i < x.M * x.N; i++)
                result[i] = Simplify(Sigmoid(Simplify(x[i])));
            return result;
        }

        private static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        private static double Simplify(double x)
        {
            return Math.Round(x * 100) / 100;
        }
    }
}
