package sdt.ch5;

public final class FocalNode {
    private boolean leaf;
    private int predictedLabel;
    private int featureIndex;
    private double threshold;
    private int neighborhoodSize;
    private FocalNode left;
    private FocalNode right;
    private int depth;
    private int size;
    private double informationGain;

    public boolean isLeaf() { return leaf; }
    public int getPredictedLabel() { return predictedLabel; }
    public int getFeatureIndex() { return featureIndex; }
    public double getThreshold() { return threshold; }
    public int getNeighborhoodSize() { return neighborhoodSize; }
    public FocalNode getLeft() { return left; }
    public FocalNode getRight() { return right; }
    public int getDepth() { return depth; }
    public int getSize() { return size; }
    public double getInformationGain() { return informationGain; }

    public void setLeaf(boolean leaf) { this.leaf = leaf; }
    public void setPredictedLabel(int predictedLabel) { this.predictedLabel = predictedLabel; }
    public void setFeatureIndex(int featureIndex) { this.featureIndex = featureIndex; }
    public void setThreshold(double threshold) { this.threshold = threshold; }
    public void setNeighborhoodSize(int neighborhoodSize) { this.neighborhoodSize = neighborhoodSize; }
    public void setLeft(FocalNode left) { this.left = left; }
    public void setRight(FocalNode right) { this.right = right; }
    public void setDepth(int depth) { this.depth = depth; }
    public void setSize(int size) { this.size = size; }
    public void setInformationGain(double informationGain) { this.informationGain = informationGain; }
}
