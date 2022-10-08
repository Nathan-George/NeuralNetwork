public class Layer {

    private double[] input;
    private double[] output;
    private double[] valueAfterBiases;

    private double[][] weights;
    private double[] biases;

    private double[][] dWeights;
    private double[] dBiases;

    int numInBatch;

    // setup with values
    public Layer(double[][] inWeights, double[] inBiases)
    {
        if (inWeights.length != inBiases.length)
        {
            throw new IllegalArgumentException("weight and biases lengths do not match");
        }

        input = new double[inWeights[0].length];
        output = new double[inBiases.length];
        valueAfterBiases = new double[inBiases.length];

        weights = new double[inWeights.length][inWeights[0].length];
        biases = new double[inBiases.length];

        for (int row = 0; row < inWeights.length; row ++)
        {
            for (int col = 0; col < inWeights[row].length; col ++)
            {
                weights[row][col] = inWeights[row][col];
            }
        }

        for (int row = 0; row < inBiases.length; row ++)
        {
            biases[row] = inBiases[row];
        }

        dWeights = new double[inWeights.length][inWeights[0].length];
        dBiases = new double[inBiases.length];

        numInBatch = 0;
    }

    // randomizes values
    public Layer(int inSize, int outSize)
    {
        input = new double[inSize];
        output = new double[outSize];
        valueAfterBiases = new double[outSize];

        weights = new double[outSize][inSize];
        biases = new double[outSize];

        for (int row = 0; row < outSize; row ++)
        {
            for (int col = 0; col < inSize; col ++)
            {
                weights[row][col] = getRandomWeight();
            }
        }

        for (int row = 0; row < outSize; row ++)
        {
            biases[row] = getRandomBias();
        }

        dWeights = new double[outSize][inSize];
        dBiases = new double[outSize];

        numInBatch = 0;
    }

    /**
     * evaluates a given input by multiplying weights, adding biases, and an activation function
     * @param in
     * @return the evaluation
     */
    public double[] evaluate(double[] in)
    {
        if (in.length != weights[0].length)
        {
            throw new IllegalArgumentException("evaluate input length must match layer input length");
        }

        input = in;
        valueAfterBiases = LinearAlgebra.add(LinearAlgebra.dot(weights, in), biases);

        return output = sigmoid(valueAfterBiases);
    }

    /**
     * dTrain updates the layer along the derivative of the output
     * @param dOut the derivative of the output of the layer with respect to the output of the network
     * @return the derivate of the input of the layer with respect to the output of the netword
     */
    public double[] dTrain(double[] dOut)
    {
        if(dOut.length != biases.length)
        {
            throw new IllegalArgumentException("dTrain input length must match layer output length");
        }

        double[] dValueAfterBiases = dSigmoid(valueAfterBiases);
        
        // calculate dWeights
        for (int row = 0; row < output.length; row++)
        {
            for (int col = 0; col < input.length; col++)
            {
                dWeights[row][col] += input[col] * dValueAfterBiases[row] * dOut[row];
            }
        }

        // calculate dBiases
        for (int i = 0; i < output.length; i++)
        {
            dBiases[i] += dValueAfterBiases[i] * dOut[i];
        }

        double[] dIn = new double[input.length];

        // calculate dIn
        for (int row = 0; row < output.length; row++)
        {
            for (int col = 0; col < input.length; col++)
            {
                dIn[col] += weights[row][col] * dValueAfterBiases[row] * dOut[row];
            }
        }

        numInBatch++;

        return dIn;
    }

    public void applyGradient(double stepSize)
    {
        // apply to weights
        for (int row = 0; row < weights.length; row ++)
        {
            for (int col = 0; col < weights[row].length; col ++)
            {
                weights[row][col] += -dWeights[row][col] / numInBatch * stepSize;
            }
        }

        // apply to biases
        for (int i = 0; i < biases.length; i ++)
        {
            biases[i] += -dBiases[i] / numInBatch * stepSize;
        }

        dWeights = new double[output.length][input.length];
        dBiases = new double[output.length];

        numInBatch = 0;
    }

    private double[] sigmoid(double[] v)
    {
        double[] vOut = new double[v.length];
        for (int i = 0; i < v.length; i++) vOut[i] = sigmoid(v[i]);
        return vOut;
    }

    private double[] dSigmoid(double[] v)
    {
        double[] vOut = new double[v.length];
        for (int i = 0; i < v.length; i++) vOut[i] = dSigmoid(v[i]);
        return vOut;
    }

    // default is ReLU
    private double sigmoid(double n)
    {
        return n < 0 ? 0.05 * n : n;
    }

    private double dSigmoid(double n)
    {
        return n < 0 ? 0.05 : 1;
    }

    private double getRandomWeight()
    {
        return Math.random() * 2 - 1;
    }

    private double getRandomBias()
    {
        return Math.random() * 2 - 1;
    }

    public void setWeightsAndBiases(double[][] inWeights, double[] inBiases)
    {
        if (inWeights.length != inBiases.length)
        {
            throw new IllegalArgumentException("weight and biases lengths do not match");
        }

        weights = new double[inWeights.length][inWeights[0].length];
        biases = new double[inBiases.length];

        for (int row = 0; row < inWeights.length; row ++)
        {
            for (int col = 0; col < inWeights[row].length; col ++)
            {
                weights[row][col] = inWeights[row][col];
            }
        }

        for (int row = 0; row < inBiases.length; row ++)
        {
            biases[row] = inBiases[row];
        }

        dWeights = new double[inWeights.length][inWeights[0].length];
        dBiases = new double[inBiases.length];
    }

    public double[][] getWeights()
    {
        double[][] weightsCopy = new double[weights.length][];
        for (int row = 0; row < weights.length; row ++)
        {
            weightsCopy[row] = new double[weights[row].length];
            for (int col = 0; col < weights[row].length; col ++)
            {
                weightsCopy[row][col] = weights[row][col];
            }
        }
        return weightsCopy;
    }

    public double[] getBiases()
    {
        double[] biasesCopy = new double[biases.length];
        for (int row = 0; row < weights.length; row ++)
        {
            biasesCopy[row] = biases[row];
        }
        return biasesCopy;
    }
}