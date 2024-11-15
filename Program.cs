using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Data;
using System.Net;

namespace HandWritingReader
{
    public class Program
    {
        public const string TRAIN_IMAGE = "train-images.idx3-ubyte";
        public const string TRAIN_LABEL = "train-labels.idx1-ubyte";
        public const string TEST_IMAGE = "t10k-images.idx3-ubyte";
        public const string TEST_LABEL = "t10k-labels.idx1-ubyte";

        public static readonly string nnPath = @"H:\ai\num_data_50.nn";
        public static readonly string imagePath = @"H:\ai\testImage.png";

        public static NeuralNetwork.TrainingData[] trainData;
        public static NeuralNetwork.TrainingData[] testData;

        static void Main(string[] args)
        {
            trainData = NeuralNetwork.ToData(MnistReader.Read("H:\\ai", TRAIN_IMAGE, TRAIN_LABEL));
            testData = NeuralNetwork.ToData(MnistReader.Read("H:\\ai", TEST_IMAGE, TEST_LABEL));

            //EpochTest();
            NeuralNetwork network = Load();
            //NeuralNetwork network = Create();

            //Test(network);

            //MonoTest(network);
            MultiTest(network);

            //SimpleNeuralNetwork nn = new SimpleNeuralNetwork(nnPath);
            //MultiTest(nn);
        }

        public static void SimpleTest(SimpleNeuralNetwork network)
        {
            while (true)
            {
                Image<Rgba32> image = Image.Load<Rgba32>(imagePath);
                double[] input = Preprocess(image.ToArray(), true, out _);
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                        Console.Write("□▤▦■"[(int)(input[j + i * 28] * 3)]);
                    Console.WriteLine();
                }
                Console.WriteLine();
                double[] output = network.FeedForward(input);
                Console.WriteLine("예측: " + Utils.MaxIndex<double>(output));
                Console.ReadLine();
            }
        }

        public static void MnistTest()
        {
            var test = MnistReader.Read("H:\\ai", TEST_IMAGE, TEST_LABEL);
            for (int i = 0; i < 20; i++)
            {
                Image<Rgba32> image = new Image<Rgba32>(28, 28);
                image.ProcessPixelRows(rows =>
                {
                    for (int j = 0; j < 28; j++)
                    {
                        var row = rows.GetRowSpan(j);
                        for (int k = 0; k < 28; k++)
                        {
                            float color = test.images[i][j * 28 + k] / 255f;
                            row[k] = new Rgba32(color, color, color);
                        }
                    }
                });
                image.SaveAsPng($@"E:\mnist\sample{i}.png");
            }
        }

        public static NeuralNetwork Load()
        {
            return new NeuralNetwork(nnPath);
        }

        public static NeuralNetwork Create()
        {
            NeuralNetwork network = new NeuralNetwork(28 * 28, 16, 16, 10);
            network.SGD(trainData, 50, 10, 3);
            try
            {
                network.Save(nnPath);
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }
            return network;
        }

        public static NeuralNetwork EpochTest()
        {
            NeuralNetwork network = new NeuralNetwork(28 * 28, 16, 16, 10);
            double max = 0;
            double epoch = 0;
            for (int i = 0; i < 50; i++)
            {
                network.SGD(trainData, 1, 10, 3);
                Console.Write($"epoch {i+1}: ");
                double p = Test(network);
                if (p > max)
                {
                    max = p;
                    epoch = i + 1;
                }
            }
            Console.WriteLine($"\nbest was epoch {epoch}, which was {(float)max * 100}%");
            Console.ReadLine();
            return network;
        }

        public static double Test(NeuralNetwork network)
        {
            var data = testData;
            int count = data.Length;
            Console.WriteLine(count);
            Console.Read();
            int pass = 0;
            for (int i = 0; i < data.Length; i++)
            {
                double[] output = network.FeedForward(data[i].input);
                if (Utils.MaxIndex<double>(output) == Utils.MaxIndex<double>(data[i].output))
                    pass++;
            }
            Console.WriteLine($"{pass} / {count} ({100f * pass / count}%)");
            return 1d * pass / count;
        }

        public static void MonoTest(NeuralNetwork network)
        {
            while (true)
            {
                Image<Rgba32> image = Image.Load<Rgba32>(imagePath);
                double[] input = Preprocess(image.ToArray(), true, out _);
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                        Console.Write("□▤▦■"[(int)(input[j + i * 28] * 3)]);
                    Console.WriteLine();
                }
                Console.WriteLine();
                double[] output = network.FeedForward(input);
                Console.WriteLine("예측: " + Utils.MaxIndex<double>(output));
                Console.ReadLine();
            }
        }

        public static void MultiTest(NeuralNetwork network)
        {
            while (true)
            {
                Console.Clear();
                Image<Rgba32> image = Image.Load<Rgba32>(imagePath);
                int[,] arr = image.ToArray(50);
                int h = image.Height, w = image.Width;
                var regions = new List<bool[,]>();
                bool[,] check = new bool[h, w];
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                    {
                        if (check[i, j] || arr[i, j] == 0)
                            continue;
                        var region = FindRegion(arr, i, j, check);
                        if (region.count != 0)
                            regions.Add(region.region);
                    }
                int[][,] numbers = new int[regions.Count][,];
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                        for (int k = 0; k < regions.Count; k++)
                            if (regions[k][i, j])
                            {
                                if (numbers[k] == null)
                                    numbers[k] = new int[h, w];
                                numbers[k][i, j] = arr[i, j];
                                break;
                            }
                var result = new (int x, int n, double[] input)[numbers.Length];
                double[][] inputs = new double[numbers.Length][];
                for (int i = 0; i < numbers.Length; i++)
                {
                    double[] input = Preprocess(numbers[i], false, out var info);
                    double[] output = network.FeedForward(input);
                    result[i] = (info.cx, Utils.MaxIndex<double>(output), input);
                }
                result = result.OrderBy(t => t.x).ToArray();

                int row = 4;
                for (int k = 0; k < 1d * result.Length / row; k++)
                {
                    for (int i = 0; i < 28; i++)
                    {
                        for (int l = 0; l < row; l++)
                        {
                            for (int j = 0; j < 28 && row * k + l < result.Length; j++)
                                Console.Write("□▤▦■"[(int)(result[row * k + l].input[j + i * 28] * 3)]);
                            Console.Write("  ");
                        }
                        Console.WriteLine();
                    }
                    Console.WriteLine();
                    Console.WriteLine();
                    Console.WriteLine();
                }
                Console.Write("예측: ");
                for (int i = 0; i < result.Length; i++)
                {
                    Console.Write(result[i].n);
                }
                Console.WriteLine();
                Console.ReadLine();
            }
        }

        private static readonly int[][] offset = [[0, 1], [-1, 0], [0, -1], [1, 0]];

        public static (int count, bool[,] region) FindRegion(int[,] arr, int y, int x, bool[,] check)
        {
            int h = arr.GetLength(0), w = arr.GetLength(1);
            int count = 0;
            var region = new bool[h, w];
            void Find(int i, int j, int dir)
            {
                if (i < 0 || i >= h || j < 0 || j >= w || arr[i, j] == 0 || region[i, j])
                    return;
                count++;
                region[i, j] = true;
                check[i, j] = true;
                for (int k = 0; k < 4; k++)
                    if ((dir + 2) % 4 != k)
                        Find(i + offset[k][0], j + offset[k][1], k);
            }
            for (int i = 0; i < 4; i++)
                Find(y + offset[i][0], x + offset[i][1], i);
            return (count, region);
        }

        public static double[] Preprocess(int[,] image, bool filter, out (int minX, int minY, int maxX, int maxY, int cx, int cy) info)
        {
            int h = image.GetLength(0), w = image.GetLength(1);

            if (filter)
            {
                int maxSize = 0;
                bool[,] check = new bool[h, w];
                bool[,] region = new bool[h, w];
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                    {
                        if (check[i, j] || image[i, j] == 0)
                            continue;
                        var find = FindRegion(image, i, j, check);
                        if (maxSize >= find.count)
                            continue;
                        maxSize = find.count;
                        region = find.region;
                    }
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                        if (!region[i, j])
                            image[i, j] = 0;
            }

            int minX = w, minY = h, maxX = 0, maxY = 0, cx = 0, cy = 0, sum = 0;
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    int a = image[i, j];
                    if (a == 0)
                        continue;
                    cx += j * a;
                    cy += i * a;
                    sum += a;
                    if (minX > j) minX = j;
                    if (minY > i) minY = i;
                    if (maxX < j) maxX = j;
                    if (maxY < i) maxY = i;
                }
            if (sum == 0)
                sum = 1;
            cx /= sum;
            cy /= sum;
            info = (minX, minY, maxX, maxY, cx, cy);

            int r = (new int[] { Math.Abs(minX - cx), Math.Abs(minY - cy), Math.Abs(maxX - cx), Math.Abs(maxY - cy) }).Max();
            int offX = cx - r, offY = cy - r;

            double maxOpacity = 0;
            double[,] resized = new double[28, 28];
            int frame = 24;
            for (int i = 0; i < frame; i++)
            {
                for (int j = 0; j < frame; j++)
                {
                    double add = 0;
                    for (int k = 0; k <= 2 * r / frame; k++)
                        for (int l = 0; l <= 2 * r / frame; l++)
                        {
                            int y = i * 2 * r / frame + k + offY, x = j * 2 * r / frame + l + offX;
                            if (y >= 0 && y < h && x >= 0 && x < w)
                                add += image[y, x];
                        }
                    resized[i + (28 - frame) / 2, j + (28 - frame) / 2] = add;
                    if (maxOpacity < add)
                        maxOpacity = add;
                }
            }

            double[] result = new double[28 * 28];
            if (maxOpacity == 0)
                return result;
            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                    result[i * 28 + j] = resized[i, j] / maxOpacity;
            return result;
        }
    }
}
