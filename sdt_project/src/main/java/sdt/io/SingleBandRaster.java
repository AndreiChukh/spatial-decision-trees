package sdt.io;

public final class SingleBandRaster {
    private final String name;
    private final int width;
    private final int height;
    private final int noData;
    private final int[] values;

    public SingleBandRaster(String name, int width, int height, int noData, int[] values) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("width and height must be positive");
        }
        if (values == null || values.length != width * height) {
            throw new IllegalArgumentException("values length must be width * height");
        }
        this.name = name;
        this.width = width;
        this.height = height;
        this.noData = noData;
        this.values = values.clone();
    }

    public String name() { return name; }
    public int width() { return width; }
    public int height() { return height; }
    public int noData() { return noData; }

    public int get(int row, int col) {
        checkCell(row, col);
        return values[row * width + col];
    }

    public boolean isNoData(int row, int col) {
        return get(row, col) == noData;
    }

    public RasterStatistics statistics() {
        long validCount = 0;
        long noDataCount = 0;
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        double mean = 0.0;
        double m2 = 0.0;

        for (int value : values) {
            if (value == noData) {
                noDataCount++;
                continue;
            }
            validCount++;
            min = Math.min(min, value);
            max = Math.max(max, value);
            double delta = value - mean;
            mean += delta / validCount;
            double delta2 = value - mean;
            m2 += delta * delta2;
        }
        if (validCount == 0) {
            throw new IllegalStateException("Raster contains no valid pixels: " + name);
        }
        double variance = validCount > 1 ? m2 / validCount : 0.0;
        return new RasterStatistics(name, width, height, validCount, noDataCount, min, max, mean, Math.sqrt(variance));
    }

    private void checkCell(int row, int col) {
        if (row < 0 || row >= height || col < 0 || col >= width) {
            throw new IndexOutOfBoundsException("Cell outside raster: (" + row + ", " + col + ")");
        }
    }
}
