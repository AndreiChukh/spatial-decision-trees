package sdt.io;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferUShort;
import java.awt.image.Raster;
import java.io.IOException;
import java.nio.file.Path;

public final class TiffReader {
    private TiffReader() {
    }

    public static SingleBandRaster readSingleBand(Path path, int noData) throws IOException {
        BufferedImage image = ImageIO.read(path.toFile());
        if (image == null) {
            throw new IOException("ImageIO could not read TIFF: " + path);
        }
        Raster raster = image.getData();
        if (raster.getNumBands() != 1) {
            throw new IOException("Expected 1 band, found " + raster.getNumBands() + " in " + path);
        }
        int width = raster.getWidth();
        int height = raster.getHeight();
        int[] values = new int[width * height];
        DataBuffer buffer = raster.getDataBuffer();

        if (buffer instanceof DataBufferUShort ushortBuffer) {
            short[] raw = ushortBuffer.getData();
            for (int i = 0; i < values.length; i++) {
                values[i] = Short.toUnsignedInt(raw[i]);
            }
        } else if (buffer instanceof DataBufferByte byteBuffer) {
            byte[] raw = byteBuffer.getData();
            for (int i = 0; i < values.length; i++) {
                values[i] = Byte.toUnsignedInt(raw[i]);
            }
        } else {
            int[] samples = raster.getSamples(0, 0, width, height, 0, (int[]) null);
            System.arraycopy(samples, 0, values, 0, samples.length);
        }
        return new SingleBandRaster(path.getFileName().toString(), width, height, noData, values);
    }
}
