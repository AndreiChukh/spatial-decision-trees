package sdt.model;

import java.util.Arrays;

public final class Node {

    private boolean leaf;
    private int predictedLabel;
    private int featureIndex;
    private double threshold;
    private Node left;
    private Node right;
    private int depth;
    private int size;
    private double informationGain;
    private double nsar;
    private double sig;
    private double[] classWeights;
    private double[] branchWeights;

    public boolean isLeaf() {
        return leaf;
    }

    public int getPredictedLabel() {
        return predictedLabel;
    }

    public int getFeatureIndex() {
        return featureIndex;
    }

    public double getThreshold() {
        return threshold;
    }

    public Node getLeft() {
        return left;
    }

    public Node getRight() {
        return right;
    }

    public int getDepth() {
        return depth;
    }

    public int getSize() {
        return size;
    }

    public double getInformationGain() {
        return informationGain;
    }

    public double getNsar() {
        return nsar;
    }

    public double getSig() {
        return sig;
    }

    public double[] getClassWeights() {
        return classWeights == null ? null : Arrays.copyOf(classWeights, classWeights.length);
    }

    public double[] getBranchWeights() {
        return branchWeights == null ? null : Arrays.copyOf(branchWeights, branchWeights.length);
    }

    public void setLeaf(boolean leaf) {
        this.leaf = leaf;
    }

    public void setPredictedLabel(int predictedLabel) {
        this.predictedLabel = predictedLabel;
    }

    public void setFeatureIndex(int featureIndex) {
        this.featureIndex = featureIndex;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public void setLeft(Node left) {
        this.left = left;
    }

    public void setRight(Node right) {
        this.right = right;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public void setInformationGain(double informationGain) {
        this.informationGain = informationGain;
    }

    public void setNsar(double nsar) {
        this.nsar = nsar;
    }

    public void setSig(double sig) {
        this.sig = sig;
    }

    public void setClassWeights(double[] classWeights) {
        this.classWeights = classWeights == null ? null : Arrays.copyOf(classWeights, classWeights.length);
    }

    public void setBranchWeights(double[] branchWeights) {
        this.branchWeights = branchWeights == null ? null : Arrays.copyOf(branchWeights, branchWeights.length);
    }
}
