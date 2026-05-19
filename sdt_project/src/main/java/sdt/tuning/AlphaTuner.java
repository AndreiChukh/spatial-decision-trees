package sdt.tuning;

import sdt.model.Sample;
import sdt.spatial.NeighborhoodDefinition;
import sdt.spatial.SpatialGraph;
import sdt.spatial.SpatialGraphBuilder;
import sdt.tree.SigSpatialDecisionTree;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;


public final class AlphaTuner {

    private AlphaTuner() {
    }

    public static TuningResult tune(List<Sample> trainingSamples,
                                    List<Sample> validationSamples,
                                    List<Double> candidateAlphas,
                                    int minNodeSize,
                                    NeighborhoodDefinition neighborhoodDefinition) {
        if (candidateAlphas == null || candidateAlphas.isEmpty()) {
            throw new IllegalArgumentException("candidateAlphas must not be empty");
        }

        List<EvaluationPoint> evaluations = new ArrayList<>();
        TuningResult best = null;

        long graphStarted = System.nanoTime();
        SpatialGraph trainingGraph = SpatialGraphBuilder.build(trainingSamples, neighborhoodDefinition);
        double graphMillis = (System.nanoTime() - graphStarted) / 1_000_000.0;
        System.out.printf(Locale.US, "  built training spatial graph once: samples=%d, time=%.1f ms%n",
                trainingGraph.size(), graphMillis);

        final double RELATIVE_THRESHOLD = 0.01;

        for (int i = 0; i < candidateAlphas.size(); i++) {
            double alpha = candidateAlphas.get(i);
            System.out.printf(Locale.US, "  checking alpha %2d/%2d: %.2f ...%n",
                    i + 1, candidateAlphas.size(), alpha);

            long started = System.nanoTime();
            SigSpatialDecisionTree tree = new SigSpatialDecisionTree(alpha, minNodeSize, neighborhoodDefinition);
            tree.fit(trainingGraph);

            double trainingError = tree.errorRate(trainingSamples);
            double validationError = tree.errorRate(validationSamples);
            double elapsedMillis = (System.nanoTime() - started) / 1_000_000.0;
            evaluations.add(new EvaluationPoint(alpha, trainingError, validationError));
            System.out.printf(Locale.US, "    result: trainError=%.4f validationError=%.4f, time=%.1f ms%n",
                    trainingError, validationError, elapsedMillis);

            // Update best only if strictly better by at least RELATIVE_THRESHOLD.
            // Ties (< 1% relative difference) keep the earlier smaller alpha.
            boolean strictlyBetter = best == null
                    || validationError < best.validationError() * (1.0 - RELATIVE_THRESHOLD);
            if (strictlyBetter) {
                best = new TuningResult(alpha, trainingError, validationError, List.copyOf(evaluations));
            }
        }

        return new TuningResult(best.alpha(), best.trainingError(), best.validationError(), List.copyOf(evaluations));
    }

    public record EvaluationPoint(double alpha, double trainingError, double validationError) {
    }

    public record TuningResult(double alpha,
                               double trainingError,
                               double validationError,
                               List<EvaluationPoint> evaluations) {
    }
}
