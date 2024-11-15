namespace HandWritingReader
{
    public class Matrix
    {
        public int M { get; private set; }
        public int N { get; private set; }

        private readonly double[] content;

        public Matrix(int m, int n)
        {
            content = new double[m * n];
            M = m;
            N = n;
        }

        public Matrix(int m, int n, Func<int, int, double> setter) : this(m, n)
        {
            if (setter != null)
            {
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        this[i, j] = setter(m, n);
            }
        }

        public static Matrix FromArray(double[,] array)
        {
            int m = array.GetLength(0), n = array.GetLength(1);
            Matrix result = new Matrix(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    result[i, j] = array[i, j];
            return result;
        }

        public static Matrix FromArray(double[] array, int axis)
        {
            if (axis == 0)
            {
                Matrix vector = new Matrix(1, array.Length);
                for (int i = 0; i < array.Length; i++)
                    vector[0, i] = array[i];
                return vector;
            }
            else if (axis == 1)
            {
                Matrix vector = new Matrix(array.Length, 1);
                for (int i = 0; i < array.Length; i++)
                    vector[i, 0] = array[i];
                return vector;
            }
            else
                throw new ArgumentException("Axis must be 0 or 1", nameof(axis));
        }

        public Matrix GetVector(int k, int axis)
        {
            if (axis == 0)
            {
                if (k < 0 || k >= M)
                    throw new IndexOutOfRangeException();
                Matrix vector = new Matrix(1, N);
                for (int i = 0; i < N; i++)
                    vector[0, i] = this[k, i];
                return vector;
            }
            else if (axis == 1)
            {
                if (k < 0 || k >= N)
                    throw new IndexOutOfRangeException();
                Matrix vector = new Matrix(M, 1);
                for (int i = 0; i < M; i++)
                    vector[i, 0] = this[i, k];
                return vector;
            }
            else
                throw new ArgumentException("Axis must be 0 or 1", nameof(axis));
        }

        public Matrix Transpose()
        {
            Matrix result = new Matrix(N, M);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    result[j, i] = this[i, j];
            return result;
        }

        public Matrix Dot(Matrix other)
        {
            if (N != other.M)
                throw new ArgumentException("Rows of a and columns of b should be same");
            Matrix result = new Matrix(M, other.N);
            for (int k = 0; k < N; k++)
                for (int i = 0; i < M; i++)
                {
                    double temp = this[i, k];
                    for (int j = 0; j < other.N; j++)
                        result[i, j] += other[k, j] * temp;
                }
            Console.WriteLine(N * M * other.N);
            return result;
        }

        public double this[int i, int j]
        {
            get => content[i * N + j];
            set => content[i * N + j] = value;
        }

        public double this[int i]
        {
            get => content[i];
            set => content[i] = value;
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.M != b.M || a.N != b.N)
                throw new ArgumentException("Matrices must have same shape");
            Matrix result = new Matrix(a.M, a.N);
            for (int i = 0; i < a.content.Length; i++)
                result.content[i] = a.content[i] + b.content[i];
            return result;
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.M != b.M || a.N != b.N)
                throw new ArgumentException("Matrices must have same shape");
            Matrix result = new Matrix(a.M, a.N);
            for (int i = 0; i < a.content.Length; i++)
                result.content[i] = a.content[i] - b.content[i];
            return result;
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            if (a.M != b.M || a.N != b.N)
                throw new ArgumentException("Matrices must have same shape");
            Matrix result = new Matrix(a.M, a.N);
            for (int i = 0; i < a.content.Length; i++)
                result.content[i] = a.content[i] * b.content[i];
            return result;
        }

        public static Matrix operator *(Matrix mat, double scl)
        {
            Matrix result = new Matrix(mat.M, mat.N);
            for (int i = 0; i < mat.content.Length; i++)
                result.content[i] = mat.content[i] * scl;
            return result;
        }

        public static Matrix operator *(double scl, Matrix mat) => mat * scl;

        public static Matrix operator /(Matrix mat, double scl) => mat * (1 / scl);

        public static Matrix operator -(Matrix mat) => mat * (-1);

        public static explicit operator double[,](Matrix mat)
        {
            double[,] result = new double[mat.M, mat.N];
            for (int i = 0; i < mat.M; i++)
                for (int j = 0; j < mat.N; j++)
                    result[i, j] = mat[i, j];
            return result;
        }

        public static explicit operator double[](Matrix mat)
        {
            if (mat.M == 1)
            {
                double[] result = new double[mat.N];
                for (int i = 0; i < mat.N; i++)
                    result[i] = mat[0, i];
                return result;
            }
            else if (mat.N == 1)
            {
                double[] result = new double[mat.M];
                for (int i = 0; i < mat.M; i++)
                    result[i] = mat[i, 0];
                return result;
            }
            else
                throw new InvalidCastException("This matrix is not a vector");
        }

        public override string ToString()
        {
            string result = "";
            for (int i = 0; i < M; i++)
            {
                result += "[";
                for (int j = 0; j < N; j++)
                {
                    result += this[i, j];
                    if (j != N - 1)
                        result += ",";
                }
                result += "]";
                if (i != M - 1)
                    result += "\n";
            }
            return result;
        }
    }
}
