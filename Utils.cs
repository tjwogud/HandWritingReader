using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace HandWritingReader
{
    public static class Utils
    {
        public static double GaussianRand(this Random rand, double avg, double sigma)
        {
            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2d * Math.Log(u1)) * Math.Sin(2d * Math.PI * u2);
            double randNormal = avg + sigma * randStdNormal;
            return randNormal;
        }

        public static int MaxIndex<T>(ReadOnlySpan<T> span) where T : IComparable<T>
        {
            int index = 0;
            for (int i = 1; i < span.Length; i++)
                if (span[index].CompareTo(span[i]) < 0)
                    index = i;
            return index;
        }

        public static int[,] ToArray(this Image<Rgba32> image, int min = -1)
        {
            int[,] result = new int[image.Height, image.Width];
            image.ProcessPixelRows(rows =>
            {
                for (int i = 0; i < rows.Height; i++)
                {
                    Span<Rgba32> row = rows.GetRowSpan(i);
                    for (int j = 0; j < rows.Width; j++)
                        if (row[j].A >= min)
                            result[i, j] = row[j].A;
                }
            });
            return result;
        }
    }
}
