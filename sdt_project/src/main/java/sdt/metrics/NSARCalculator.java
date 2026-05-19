package sdt.metrics;

import sdt.model.Sample;
import sdt.spatial.SpatialGraph;


public final class NSARCalculator {

    private NSARCalculator() {
    }

    public static double calculateAverageNSAR(SpatialGraph parent,
                                              SpatialGraph left,
                                              SpatialGraph right) {
        if (parent.size() == 0) {
            return 0.0;
        }

        double sum = 0.0;
        for (Sample sample : parent.samples()) {
            int before = homogeneousNeighborCount(parent, sample);
            int after = 0;

            if (left.contains(sample.getId())) {
                after = homogeneousNeighborCount(left, sample);
            } else if (right.contains(sample.getId())) {
                after = homogeneousNeighborCount(right, sample);
            }

            double nsar = before == 0 ? 1.0 : ((double) after / before);
            sum += nsar;
        }

        return sum / parent.size();
    }

    private static int homogeneousNeighborCount(SpatialGraph graph, Sample sample) {
        int count = 0;
        for (Sample neighbor : graph.neighborsOf(sample.getId())) {
            if (neighbor.getLabel() == sample.getLabel()) {
                count++;
            }
        }
        return count;
    }
}
