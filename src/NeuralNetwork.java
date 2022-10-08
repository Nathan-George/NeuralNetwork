import java.io.*;

public class NeuralNetwork {
    
    private int numLayers;
    private int[] layerSizes;

    private Layer[] layers;

    // training vars
    private int batchSize;
    private double stepSize;

    private int numInBatch;

    public NeuralNetwork(File inFile)
    {
        setUpFromFile(inFile);

        // setup training vars
        batchSize = 1;
        stepSize = 0.02;
        numInBatch = 0;
    }

    public NeuralNetwork(int[] inLayerSizes, File inFile)
    {
        numLayers = inLayerSizes.length - 1;

        layerSizes = new int[inLayerSizes.length];

        for (int layer = 0; layer < inLayerSizes.length; layer ++)
        {
            layerSizes[layer] = inLayerSizes[layer];
        }

        layers = new Layer[numLayers];

        for (int layer = 0; layer < numLayers; layer ++)
        {
            layers[layer] = new Layer(layerSizes[layer], layerSizes[layer + 1]);
        }

        saveToFile(inFile);

        // setup training vars
        batchSize = 1;
        stepSize = 0.02;
        numInBatch = 0;
    }

    private void setUpFromFile(File saveFile)
    {
        try (FileInputStream fileReader = new FileInputStream(saveFile))
        {
            // get num layers
            numLayers = fileReader.read();

            layerSizes = new int[numLayers + 1];
            
            // get layer sizes
            for (int layer = 0; layer < numLayers + 1; layer ++)
            {
                int layerSize = 0;
                for (int i = 0; i < 4; i++) layerSize |= ((int) fileReader.read() & 0xFF) << (i * 8);
                layerSizes[layer] = layerSize;
            }

            layers = new Layer[numLayers];

            for (int layer = 0; layer < numLayers; layer++)
            {
                double[][] weights = new double[layerSizes[layer + 1]][layerSizes[layer]];

                // get weights from file
                for (int row = 0; row < layerSizes[layer + 1]; row++)
                {
                    for (int col = 0; col < layerSizes[layer]; col++)
                    {
                        long rawWeight = 0;
                        for (int i = 0; i < 8; i++) rawWeight |= ((long) fileReader.read() & 0xFF) << (i * 8);
                        weights[row][col] = Double.longBitsToDouble(rawWeight);

                    }
                }

                double[] biases = new double[layerSizes[layer + 1]];

                // get biases from file
                for (int row = 0; row < layerSizes[layer + 1]; row++)
                {
                    long rawBias = 0;
                    for (int i = 0; i < 8; i++) rawBias |= ((long) fileReader.read() & 0xFF) << (i * 8);
                    biases[row] = Double.longBitsToDouble(rawBias);
                }

                layers[layer] = new Layer(weights, biases);
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public void saveToFile(File saveFile)
    {
        try (FileOutputStream fileWriter = new FileOutputStream(saveFile))
        {
            fileWriter.write(numLayers);
            
            for  (int layerSize : layerSizes)
            {
                for (int i = 0; i < 4; i ++)
                {
                    fileWriter.write((byte) layerSize);

                    layerSize >>>= 8;
                }
            }

            for  (Layer layer : layers)
            {
                double[][] weights = layer.getWeights();

                for (double[] row : weights)
                {
                    for (double weight : row)
                    {
                        long rawWeight = Double.doubleToLongBits(weight);
                        for (int i = 0; i < 8; i ++)
                        {
                            fileWriter.write((byte) rawWeight);

                            rawWeight >>>= 8;
                        }
                    }
                }

                double[] biases = layer.getBiases();

                for (double bias : biases)
                {
                    long rawBias = Double.doubleToLongBits(bias);
                    for (int i = 0; i < 8; i ++)
                    {
                        fileWriter.write((byte) rawBias);

                        rawBias >>>= 8;
                    }
                }
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    /**
     * evaluates a given input to the neural network
     * 
     * @param in
     * @return the evaluation of the network
     */
    public double[] evaluate(double[] in)
    {
        double[] out = in;
        
        for (Layer layer : layers)
        {
            out = layer.evaluate(out);
        }

        return out;
    }

    /**
     * train uses a general cost fuction to update the network to match a given input to a given output
     * 
     * @param in is the the given input to the neural network
     * @param target is the target output of the neural network
     * @return the derivative of the fuction with respect to the input of the network
     */
    public double[] train(double[] in, double[] target)
    {
        if (in.length != layerSizes[0])
        {
            throw new IllegalArgumentException("in length must match in length of the first layer of the network");
        }

        if (target.length != layerSizes[numLayers])
        {
            throw new IllegalArgumentException("out length must match out length of the last layer in the netword");
        }

        double[] netOut = evaluate(in);

        double[] dCost = new double[netOut.length];

        for (int i = 0; i < netOut.length; i ++)
        {
            dCost[i] = 2 * (netOut[i] - target[i]);
        }

        return dTrain(dCost);
    }

    /**
     * Must have a called evaluate previously
     * @param dOut is the derivative of the output of the network (mostly used with the cost function)
     * @return the derivative with respect to the input of the last call to evaluate
     */
    public double[] dTrain(double[] dOut)
    {
        for (int layer = numLayers - 1; layer >= 0; layer --)
        {
            dOut = layers[layer].dTrain(dOut);
        }

        if (++numInBatch >= batchSize)
        {
            for (Layer layer : layers)
            {
                layer.applyGradient(stepSize);
            }

            numInBatch = 0;
        }

        return dOut;
    }

    public void setBatchSize(int inBatchSize)
    {
        batchSize = inBatchSize;
    }

    public void setStepSize(double inStepSize)
    {
        stepSize = inStepSize;
    }

    public Layer[] getLayers()
    {
        return layers;
    }

    public int[] getLayerSizes()
    {
        return layerSizes;
    }

    public int getNumLayers()
    {
        return numLayers;
    }
}
