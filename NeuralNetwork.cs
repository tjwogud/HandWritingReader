using System.Collections.Generic;

namespace HandWritingReader
{
    public class NeuralNetwork
    {
        public struct TrainingData
        {
            public double[] input;
            public double[] output;

            public TrainingData(double[] input, double[] output)
            {
                this.input = input;
                this.output = output;
            }
        }

        private int[] sizes;
        private Matrix[] biases;
        private Matrix[] weights;

        public NeuralNetwork(params int[] sizes)
        {
            if (sizes.Length <= 1)
                throw new ArgumentException("Levels should be more than 1");
            this.sizes = sizes;
            biases = new Matrix[sizes.Length - 1];
            weights = new Matrix[sizes.Length - 1];

            Random rand = new Random();
            for (int i = 0; i < sizes.Length - 1; i++)
            {
                biases[i] = new Matrix(sizes[i + 1], 1, (m, n) => rand.GaussianRand(0, 1));
                weights[i] = new Matrix(sizes[i + 1], sizes[i], (m, n) => rand.GaussianRand(0, 1));
            }
        }

        public NeuralNetwork(string path)
        {
            using FileStream fs = new FileStream(path, FileMode.Open);
            using BinaryReader bw = new BinaryReader(fs);
            sizes = new int[bw.ReadInt32()];
            for (int i = 0; i < sizes.Length; i++)
                sizes[i] = bw.ReadInt32();
            biases = new Matrix[sizes.Length - 1];
            for (int i = 0; i < sizes.Length - 1; i++)
                biases[i] = new Matrix(sizes[i + 1], 1, (m, n) => bw.ReadDouble());
            weights = new Matrix[sizes.Length - 1];
            for (int i = 0; i < sizes.Length - 1; i++)
                weights[i] = new Matrix(sizes[i + 1], sizes[i], (m, n) => bw.ReadDouble());
        }

        public void Save(string path)
        {
            using FileStream fs = new FileStream(path, FileMode.Create);
            using BinaryWriter bw = new BinaryWriter(fs);
            bw.Write(sizes.Length);
            for (int i = 0; i < sizes.Length; i++)
                bw.Write(sizes[i]);
            for (int i = 0; i < biases.Length; i++)
                for (int j = 0; j < biases[i].M * biases[i].N; j++)
                    bw.Write(biases[i][j]);
            for (int i = 0; i < weights.Length; i++)
                for (int j = 0; j < weights[i].M * weights[i].N; j++)
                    bw.Write(weights[i][j]);
        }

        public double[] FeedForward(double[] input)
        {
            Matrix a = Matrix.FromArray(input, 1);
            for (int i = 0; i < sizes.Length - 1; i++)
            {
                a = Activation(weights[i].Dot(a) + biases[i]);
            }
            return (double[])a;
        }

        public void SGD(Span<TrainingData> data, int epochs, int batchSize, double eta)
        {
            int batches = data.Length / batchSize;
            Console.WriteLine("Learning Start");
            Random rand = new Random();
            for (int i = 0; i < epochs; i++)
            {
                rand.Shuffle(data);

                for (int j = 0; j < batches; j++)
                {
                    Console.Write(i + 1);
                    Console.Write("/");
                    Console.Write(epochs);
                    Console.Write(" [");
                    for (int k = 0; k < 20; k++)
                        Console.Write(((double)j / batches) > k / 20d ? "/" : " ");
                    Console.Write("]\r");
                    LearnBatch(data.Slice(j * batchSize, batchSize), eta);
                }
            }
            Console.WriteLine("\nLearning End");
        }

        private void LearnBatch(ReadOnlySpan<TrainingData> batch, double eta)
        {
            Matrix[] nablaB = new Matrix[sizes.Length - 1];
            Matrix[] nablaW = new Matrix[sizes.Length - 1];
            for (int i = 0; i < sizes.Length - 1; i++)
            {
                nablaB[i] = new Matrix(sizes[i + 1], 1);
                nablaW[i] = new Matrix(sizes[i + 1], sizes[i]);
            }
            for (int i = 0; i < batch.Length; i++)
            {
                var result = Backprop(batch[i]);
                for (int j = 0; j < sizes.Length - 1; j++)
                {
                    nablaB[j] += result.nablaB[j];
                    nablaW[j] += result.nablaW[j];
                }
            }

            for (int i = 0; i < sizes.Length - 1; i++)
            {
                biases[i] -= eta / batch.Length * nablaB[i];
                weights[i] -= eta / batch.Length * nablaW[i];
            }
        }

        private (Matrix[] nablaB, Matrix[] nablaW) Backprop(TrainingData data)
        {
            Matrix x = Matrix.FromArray(data.input, 1);
            Matrix y = Matrix.FromArray(data.output, 1);

            Matrix[] nablaB = new Matrix[sizes.Length - 1];
            Matrix[] nablaW = new Matrix[sizes.Length - 1];

            Matrix[] acts = new Matrix[sizes.Length];
            Matrix act = x;
            acts[0] = act;
            Matrix[] zs = new Matrix[sizes.Length - 1];
            for (int i = 0; i < sizes.Length - 1; i++)
            {
                Matrix z = weights[i].Dot(act) + biases[i];
                zs[i] = z;
                act = Activation(z);
                acts[i + 1] = act;
            }

            Matrix delta = (acts[^1] - y) * ActivationPrime(zs[^1]);
            nablaB[^1] = delta;
            nablaW[^1] = delta.Dot(acts[^2].Transpose());

            for (int i = 2; i < sizes.Length; i++)
            {
                Matrix z = zs[^i];
                Matrix sp = ActivationPrime(z);
                delta = weights[^(i - 1)].Transpose().Dot(delta) * sp;
                nablaB[^i] = delta;
                nablaW[^i] = delta.Dot(acts[^(i + 1)].Transpose());
            }
            return (nablaB, nablaW);
        }

        private static Matrix Activation(Matrix x)
        {
            Matrix result = new Matrix(x.M, x.N);
            for (int i = 0; i < x.M * x.N; i++)
                result[i] = Sigmoid(x[i]);
            return result;
        }

        private static Matrix ActivationPrime(Matrix x)
        {
            Matrix result = new Matrix(x.M, x.N);
            for (int i = 0; i < x.M * x.N; i++)
                result[i] = SigmoidPrime(x[i]);
            return result;
        }

        private static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        private static double SigmoidPrime(double x)
        {
            return Sigmoid(x) * (1 - Sigmoid(x));
        }

        private static double Relu(double x)
        {
            return x > 0 ? x : 0;
        }

        private static double ReluPrime(double x)
        {
            return x > 0 ? 1 : 0;
        }

        private static Matrix Softmax(Matrix x)
        {
            Matrix result = new Matrix(x.M, x.N);
            double sum = 0;
            for (int i = 0; i < x.M * x.N; i++)
                sum += result[i] = Math.Exp(x[i]);
            return result / sum;
        }

        public static TrainingData[] ToData(Mnist mnist)
        {
            TrainingData[] result = new TrainingData[mnist.count];
            for (int i = 0; i < mnist.count; i++)
            {
                double[] input = new double[28 * 28];
                for (int j = 0; j < input.Length; j++)
                    input[j] = mnist.images[i][j] / 255d;
                double[] output = new double[10];
                output[mnist.labels[i]] = 1;
                result[i] = new TrainingData(input, output);
            }
            return result;
        }
    }
}
