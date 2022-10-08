

public class LinearAlgebra {
    public static double[] add(double[] v1, double[] v2)
    {
        if (v1.length != v2.length)
        {
            throw new IllegalArgumentException("vector dimentions do not match");
        }

        double[] sum = new double[v1.length];
        for (int i = 0; i < sum.length; i++) sum[i] = v1[i] + v2[i];
        return sum;
    }

    public static double dot(double[] v1, double[] v2)
    {
        if (v1.length != v2.length)
        {
            throw new IllegalArgumentException("vector dimentions do not match");
        }

        double out = 0;
        for (int i = 0; i < v1.length; i++) out += v1[i] * v2[i];
        return out;
    }

    /**
     * 
     * @param m is a matrix where the rows is the height and cols is the width
     * @param v is a vector whose length matches the cols of the matrix
     * @return the dot product of a matrix and a vector
     */
    public static double[] dot(double[][] m, double[] v)
    {
        if(m[0].length != v.length)
        {
            throw new IllegalArgumentException("matrix and vector dimentions do not match");
        }

        double[] vOut = new double[m.length];
        for (int i = 0; i < vOut.length; i++) vOut[i] = dot(m[i], v);
        return vOut;
    }
}
