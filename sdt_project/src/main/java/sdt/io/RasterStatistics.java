package sdt.io;

public record RasterStatistics(String name,
                               int width,
                               int height,
                               long validCount,
                               long noDataCount,
                               int min,
                               int max,
                               double mean,
                               double stddev) {
}
