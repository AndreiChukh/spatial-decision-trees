package sdt.metrics;

import sdt.spatial.SpatialGraph;

public final class InformationGainCalculator {

    private InformationGainCalculator() {
    }

    public static double calculate(double parentEntropy,
                                   SpatialGraph left,
                                   SpatialGraph right,
                                   int totalSize) {
        double leftWeight = (double) left.size() / totalSize;
        double rightWeight = (double) right.size() / totalSize;

        double childEntropy =
                leftWeight * EntropyUtils.entropy(left) +
                rightWeight * EntropyUtils.entropy(right);

        return parentEntropy - childEntropy;
    }
}
