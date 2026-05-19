package sdt.model;

import java.util.Arrays;
import java.util.Objects;


public final class Sample {
    private final int id;
    private final int row;
    private final int col;
    private final double[] features;
    private final int label;

    public Sample(int id, int row, int col, double[] features, int label) {
        this.id = id;
        this.row = row;
        this.col = col;
        this.features = Arrays.copyOf(Objects.requireNonNull(features, "features"), features.length);
        this.label = label;
    }

    public int getId() {
        return id;
    }

    public int getRow() {
        return row;
    }

    public int getCol() {
        return col;
    }

    public double[] getFeatures() {
        return Arrays.copyOf(features, features.length);
    }

    public double getFeature(int index) {
        return features[index];
    }

    public int getFeatureCount() {
        return features.length;
    }

    public int getLabel() {
        return label;
    }
}
