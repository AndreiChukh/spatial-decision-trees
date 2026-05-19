package sdt.metrics;

import sdt.model.Sample;
import sdt.spatial.SpatialGraph;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class EntropyUtils {

    private EntropyUtils() {
    }

    public static double entropy(List<Sample> samples) {
        if (samples == null || samples.isEmpty()) {
            return 0.0;
        }

        Map<Integer, Integer> counts = new HashMap<>();
        for (Sample s : samples) {
            counts.merge(s.getLabel(), 1, Integer::sum);
        }

        double n = samples.size();
        double entropy = 0.0;
        for (int count : counts.values()) {
            double p = count / n;
            if (p > 0.0) {
                entropy -= p * log2(p);
            }
        }
        return entropy;
    }

    public static double entropy(SpatialGraph graph) {
        return entropy(graph.samples());
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }
}
