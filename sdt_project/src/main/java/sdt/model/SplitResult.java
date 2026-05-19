package sdt.model;

import sdt.spatial.SpatialGraph;

public final class SplitResult {
    private int featureIndex;
    private double threshold;
    private double informationGain;
    private double nsar;
    private double sig;
    private SpatialGraph leftGraph;
    private SpatialGraph rightGraph;

    public int getFeatureIndex() {
        return featureIndex;
    }

    public void setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public double getInformationGain() {
        return informationGain;
    }

    public void setInformationGain(double informationGain) {
        this.informationGain = informationGain;
    }

    public double getNsar() {
        return nsar;
    }

    public void setNsar(double nsar) {
        this.nsar = nsar;
    }

    public double getSig() {
        return sig;
    }

    public void setSig(double sig) {
        this.sig = sig;
    }

    public SpatialGraph getLeftGraph() {
        return leftGraph;
    }

    public void setLeftGraph(SpatialGraph leftGraph) {
        this.leftGraph = leftGraph;
    }

    public SpatialGraph getRightGraph() {
        return rightGraph;
    }

    public void setRightGraph(SpatialGraph rightGraph) {
        this.rightGraph = rightGraph;
    }
}
