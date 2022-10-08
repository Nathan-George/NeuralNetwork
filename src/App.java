import java.io.File;

public class App {
    public static void main(String[] args) throws Exception {
        File networkDataFile = new File("C:\\Users\\nbg05\\OneDrive\\Documents\\Java\\NueralNetwork\\NeuralNetwork\\src\\NetworkData");
        File networkDataFileCopy = new File("C:\\Users\\nbg05\\OneDrive\\Documents\\Java\\NueralNetwork\\NeuralNetwork\\src\\NetworkDataCopy");

        File imageTrainingFile = new File("C:\\Users\\nbg05\\OneDrive\\Documents\\Java\\NueralNetwork\\NeuralNetwork\\src\\Data\\train-images.idx3-ubyte");
        File lableTrainingFile = new File("C:\\Users\\nbg05\\OneDrive\\Documents\\Java\\NueralNetwork\\NeuralNetwork\\src\\Data\\train-labels.idx1-ubyte");

        File imageTestingFile = new File("C:\\Users\\nbg05\\OneDrive\\Documents\\Java\\NueralNetwork\\NeuralNetwork\\src\\Data\\t10k-images.idx3-ubyte");
        File lableTestingFile = new File("C:\\Users\\nbg05\\OneDrive\\Documents\\Java\\NueralNetwork\\NeuralNetwork\\src\\Data\\t10k-labels.idx1-ubyte");

        MNISTImageReader mnistTraining = new MNISTImageReader(imageTrainingFile, lableTrainingFile);
        MNISTImageReader mnistTesting = new MNISTImageReader(imageTestingFile, lableTestingFile);
        

        NeuralNetwork net = new NeuralNetwork(networkDataFile);
        //NeuralNetwork net = new NeuralNetwork(new int[] {784, 50, 40, 10}, networkDataFile);

        net.setBatchSize(20);
        net.setStepSize(0.000005);

        for (int i = 0; i < 0; i++)
        {
            trainNet(net, mnistTraining, 10000);
            System.out.println("Trained on " + ((i + 1) * 10000L) + " Examples");
        }

        System.out.println("Saving to File");
        net.saveToFile(networkDataFile);
        System.out.println("Saved to File");
        System.out.println("Saving to Backup File");
        net.saveToFile(networkDataFileCopy);
        System.out.println("Saved to Backup File");

        System.out.println(testNet(net, mnistTesting));
    }

    public static void trainNet(NeuralNetwork net, MNISTImageReader mnist, long numExamples)
    {
        int currentExample = 0;

        while (currentExample < numExamples)
        {
            int exampleNum = (int) (Math.random() * mnist.getNumImages());

            int[][] image = mnist.getImage(exampleNum);
            int lable = mnist.getLable(exampleNum);

            double[] convertedImage = new double[784];
            double[] convertedLable = new double[10];

            for (int row = 0; row < 28; row ++)
            {
                for (int col = 0; col < 28; col ++)
                {
                    convertedImage[row * 28 + col] = image[row][col] / 255.0;
                }
            }

            for (int i = 0; i < 10; i ++) convertedLable[i] = 0;
            convertedLable[lable] = 1;

            net.train(convertedImage, convertedLable);

            currentExample++;
        }
    }

    public static double testNet(NeuralNetwork net, MNISTImageReader mnist, long numExamples)
    {
        double percentage = 0;

        int currentExample = 0;

        while (currentExample < numExamples)
        {
            int exampleNum = (int) (Math.random() * mnist.getNumImages());

            int[][] image = mnist.getImage(exampleNum);
            int lable = mnist.getLable(exampleNum);

            double[] convertedImage = new double[784];

            for (int row = 0; row < 28; row ++)
            {
                for (int col = 0; col < 28; col ++)
                {
                    convertedImage[row * 28 + col] = image[row][col] / 255.0;
                }
            }

            double[] rawEvaluation = net.evaluate(convertedImage);

            int evaluation = 0;
            for (int i = 1; i < 10; i++) if (rawEvaluation[i] > rawEvaluation[evaluation]) evaluation = i;

            if (evaluation == lable)
            {
                percentage += 100;
            }

            currentExample++;
        }

        return percentage / numExamples;
    }

    public static double testNet(NeuralNetwork net, MNISTImageReader mnist)
    {
        double percentage = 0;

        int currentExample = 0;

        while (currentExample < mnist.getNumImages())
        {
            int[][] image = mnist.getImage(currentExample);
            int lable = mnist.getLable(currentExample);

            double[] convertedImage = new double[784];

            for (int row = 0; row < 28; row ++)
            {
                for (int col = 0; col < 28; col ++)
                {
                    convertedImage[row * 28 + col] = image[row][col] / 255.0;
                }
            }

            double[] rawEvaluation = net.evaluate(convertedImage);

            int evaluation = 0;
            for (int i = 1; i < 10; i++) if (rawEvaluation[i] > rawEvaluation[evaluation]) evaluation = i;

            if (evaluation == lable)
            {
                percentage += 100;
            }

            currentExample++;
        }

        return percentage / mnist.getNumImages();
    }
}
