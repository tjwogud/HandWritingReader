namespace HandWritingReader
{
    public class MnistReader
    {
        public static Mnist Read(string dir, string images, string labels)
        {
            static int ToInt32(byte[] arr) => (arr[0] << 24) + (arr[1] << 16) + (arr[2] << 8) + arr[3];

            Mnist v = default;

            using FileStream fs1 = new FileStream(Path.Combine(dir, images), FileMode.Open);
            using FileStream fs2 = new FileStream(Path.Combine(dir, labels), FileMode.Open);
            byte[] buffer = new byte[4];
            fs1.Seek(4, SeekOrigin.Current);
            fs1.Read(buffer, 0, 4); v.count = ToInt32(buffer);
            fs1.Read(buffer, 0, 4); v.width = ToInt32(buffer);
            fs1.Read(buffer, 0, 4); v.height = ToInt32(buffer);
            fs2.Seek(8, SeekOrigin.Current);
            v.images = new byte[v.count][];
            v.labels = new byte[v.count];
            int len = v.width * v.height;
            for (int i = 0; i < v.count; i++)
            {
                fs1.Read(v.images[i] = new byte[len], 0, len);
                v.labels[i] = (byte)fs2.ReadByte();
            }
            return v;
        }
    }
}
