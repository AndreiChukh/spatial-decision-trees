package sdt.tree;

import sdt.model.Sample;

import java.util.List;

public interface TreeClassifier {
    int predict(double[] features);

    default double errorRate(List<Sample> samples) {
        if (samples == null || samples.isEmpty()) {
            return 0.0;
        }
        int errors = 0;
        for (Sample sample : samples) {
            if (predict(sample.getFeatures()) != sample.getLabel()) {
                errors++;
            }
        }
        return (double) errors / samples.size();
    }
}
