import java.io.*;

public class MNISTImageReader {
    private final int IMAGE_HEADING_OFFSET = 16;
    private final int IMAGE_OFFSET = 784;

    private final int LABLE_HEADING_OFFSET = 8;
    private final int LABLE_OFFSET = 1;

    File imageFile;
    File lableFile;

    private int dimentionRow;
    private int dimentionCol;

    private int imageMagicNumber;
    private int lableMagicNumber;

    private int numImages;

    public MNISTImageReader(File inImageFile, File inLableFile)
    {
        imageFile = inImageFile;
        lableFile = inLableFile;

        // get images
        try (FileInputStream imageReader = new FileInputStream(imageFile))
        {
            imageMagicNumber = getInt(imageReader);
            numImages = getInt(imageReader);
            dimentionRow = getInt(imageReader);
            dimentionCol = getInt(imageReader);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }

        // get lables
        try (FileInputStream lableReader = new FileInputStream(lableFile))
        {
            lableMagicNumber = getInt(lableReader);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }

    private static int getInt(FileInputStream reader) throws IOException
    {
        int num = 0;
        for (int i = 0; i < 4; i++) num |= (reader.read() & 0xFF) << (i * 8);
        return Integer.reverseBytes(num);
    }

    public int[][] getImage(int num)
    {
        long offset = IMAGE_HEADING_OFFSET + (long) IMAGE_OFFSET * num;

        int[][] image = new int[dimentionRow][dimentionCol];

        try (FileInputStream imageReader = new FileInputStream(imageFile))
        {
            imageReader.skip(offset);

            for (int row = 0; row < dimentionRow; row++)
            {
                for (int col = 0; col < dimentionCol; col++)
                {
                    image[row][col] = imageReader.read();
                }
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        return image;
    }

    public int getLable(int num)
    {
        long offset = LABLE_HEADING_OFFSET + (long) LABLE_OFFSET * num;

        int lable = -1;

        try (FileInputStream lableReader = new FileInputStream(lableFile))
        {
            lableReader.skip(offset);

            lable = lableReader.read();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        return lable;
    }

    public int getImageMagicNumber()
    {
        return imageMagicNumber;
    }

    public int getLableMagicNumber()
    {
        return lableMagicNumber;
    }

    public int getNumImages()
    {
        return numImages;
    }
}
