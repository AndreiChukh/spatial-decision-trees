package sdt.split;

import sdt.metrics.EntropyUtils;
import sdt.model.Sample;
import sdt.model.SplitResult;
import sdt.spatial.SpatialGraph;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public final class SplitFinder {
    private static final boolean DEBUG_NSAR = false;
    private static final double TIE_EPS = 1e-12;

    private final double alpha;
    private final int minNodeSize;

    public SplitFinder(double alpha, int minNodeSize) {
        if (alpha < 0.0 || alpha > 1.0) throw new IllegalArgumentException("alpha must be in [0, 1]");
        if (minNodeSize <= 0) throw new IllegalArgumentException("minNodeSize must be > 0");
        this.alpha = alpha;
        this.minNodeSize = minNodeSize;
    }

    public SplitResult findBestSplit(SpatialGraph graph) {
        if (graph == null || graph.size() == 0) return null;
        List<Sample> samples = graph.samples();
        int n = samples.size();
        int featureCount = samples.get(0).getFeatureCount();
        int classCount = inferClassCount(samples);
        double parentEntropy = EntropyUtils.entropy(graph);
        Map<Integer, Integer> indexById = new HashMap<>(n * 2);
        for (int i = 0; i < n; i++) indexById.put(samples.get(i).getId(), i);
        int[] beforeHomogeneous = new int[n];
        int[][] homogeneousNeighbors = buildHomogeneousNeighborIndex(graph, samples, indexById, beforeHomogeneous);

        SplitResult best = null;
        NsarStats nsarStats = new NsarStats();
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            SplitResult candidate = findBestSplitForFeature(samples, featureIndex, classCount, parentEntropy, beforeHomogeneous, homogeneousNeighbors, nsarStats);
            if (candidate != null && (best == null || candidate.getSig() > best.getSig())) best = candidate;
        }

        if (best != null) {
            List<Sample> leftSamples = new ArrayList<>();
            List<Sample> rightSamples = new ArrayList<>();
            for (Sample sample : samples) {
                if (sample.getFeature(best.getFeatureIndex()) <= best.getThreshold()) leftSamples.add(sample);
                else rightSamples.add(sample);
            }
            best.setLeftGraph(graph.inducedSubgraph(leftSamples));
            best.setRightGraph(graph.inducedSubgraph(rightSamples));

            if (DEBUG_NSAR && nsarStats.count > 0) {
                System.out.println("Node debug:");
                System.out.println("  samples = " + n);
                System.out.println("  neighbor edges = " + countUndirectedEdges(graph));
                System.out.println("  candidate splits = " + nsarStats.count);
                System.out.println("  minNSAR = " + nsarStats.min);
                System.out.println("  avgNSAR = " + (nsarStats.sum / nsarStats.count));
                System.out.println("  maxNSAR = " + nsarStats.max);
                System.out.println("  bestNSAR = " + best.getNsar());
            }
        }
        return best;
    }

    private int countUndirectedEdges(SpatialGraph graph) {
        int links = 0;
        for (Sample sample : graph.samples()) {
            links += graph.neighborIdsOf(sample.getId()).size();
        }
        return links / 2;
    }

    private SplitResult findBestSplitForFeature(List<Sample> samples,
                                                int featureIndex,
                                                int classCount,
                                                double parentEntropy,
                                                int[] beforeHomogeneous,
                                                int[][] homogeneousNeighbors,
                                                NsarStats nsarStats) {
        int n = samples.size();
        List<Integer> order = new ArrayList<>(n);
        for (int i = 0; i < n; i++) order.add(i);
        order.sort(Comparator
                .comparingDouble((Integer idx) -> samples.get(idx).getFeature(featureIndex))
                .thenComparingInt(idx -> samples.get(idx).getId()));

        int[] leftCounts = new int[classCount];
        int[] rightCounts = new int[classCount];
        for (Sample sample : samples) rightCounts[sample.getLabel()]++;

        boolean[] inLeft = new boolean[n];
        int[] afterHomogeneous = beforeHomogeneous.clone();
        double nsarSum = n;
        int leftSize = 0;
        int rightSize = n;
        SplitResult best = null;
        double bestBalance = -1.0;
        int cursor = 0;
        while (cursor < n) {
            double threshold = samples.get(order.get(cursor)).getFeature(featureIndex);
            int movedUntil = cursor;
            while (movedUntil < n && Double.compare(samples.get(order.get(movedUntil)).getFeature(featureIndex), threshold) == 0) {
                int movedIndex = order.get(movedUntil);
                int label = samples.get(movedIndex).getLabel();
                inLeft[movedIndex] = true;
                leftCounts[label]++;
                rightCounts[label]--;
                leftSize++;
                rightSize--;
                for (int neighborIndex : homogeneousNeighbors[movedIndex]) {
                    if (inLeft[neighborIndex]) {
                        nsarSum = changeAfterCount(nsarSum, movedIndex, +1, afterHomogeneous, beforeHomogeneous);
                        nsarSum = changeAfterCount(nsarSum, neighborIndex, +1, afterHomogeneous, beforeHomogeneous);
                    } else {
                        nsarSum = changeAfterCount(nsarSum, movedIndex, -1, afterHomogeneous, beforeHomogeneous);
                        nsarSum = changeAfterCount(nsarSum, neighborIndex, -1, afterHomogeneous, beforeHomogeneous);
                    }
                }
                movedUntil++;
            }
            if (leftSize >= minNodeSize && rightSize >= minNodeSize) {
                double ig = informationGain(parentEntropy, leftCounts, rightCounts, leftSize, rightSize);
                double nsar = nsarSum / n;
                nsarStats.add(nsar);
                double sig = (1.0 - alpha) * ig + alpha * nsar;
                double balance = Math.min(leftSize, rightSize) / (double) n;
                if (isBetterCandidate(best, sig, ig, balance, bestBalance)) {
                    best = new SplitResult();
                    best.setFeatureIndex(featureIndex);
                    best.setThreshold(threshold);
                    best.setInformationGain(ig);
                    best.setNsar(nsar);
                    best.setSig(sig);
                    bestBalance = balance;
                }
            }
            cursor = movedUntil;
        }
        return best;
    }


    private boolean isBetterCandidate(SplitResult best, double sig, double ig, double balance, double bestBalance) {
        if (best == null) {
            return true;
        }
        if (sig > best.getSig() + TIE_EPS) {
            return true;
        }
        if (Math.abs(sig - best.getSig()) <= TIE_EPS) {
            if (ig > best.getInformationGain() + TIE_EPS) {
                return true;
            }
            if (Math.abs(ig - best.getInformationGain()) <= TIE_EPS && balance > bestBalance + TIE_EPS) {
                return true;
            }
        }
        return false;
    }

    private int[][] buildHomogeneousNeighborIndex(SpatialGraph graph, List<Sample> samples, Map<Integer, Integer> indexById, int[] beforeHomogeneous) {
        int n = samples.size();
        List<List<Integer>> homogeneous = new ArrayList<>(n);
        for (int i = 0; i < n; i++) homogeneous.add(new ArrayList<>());
        for (int i = 0; i < n; i++) {
            Sample sample = samples.get(i);
            for (int neighborId : graph.neighborIdsOf(sample.getId())) {
                Integer j = indexById.get(neighborId);
                if (j == null || j <= i) continue;
                if (samples.get(j).getLabel() == sample.getLabel()) {
                    homogeneous.get(i).add(j);
                    homogeneous.get(j).add(i);
                    beforeHomogeneous[i]++;
                    beforeHomogeneous[j]++;
                }
            }
        }
        int[][] result = new int[n][];
        for (int i = 0; i < n; i++) {
            List<Integer> list = homogeneous.get(i);
            result[i] = new int[list.size()];
            for (int k = 0; k < list.size(); k++) result[i][k] = list.get(k);
        }
        return result;
    }

    private double changeAfterCount(double nsarSum, int sampleIndex, int delta, int[] afterHomogeneous, int[] beforeHomogeneous) {
        int before = beforeHomogeneous[sampleIndex];
        if (before == 0) return nsarSum;
        double oldContribution = (double) afterHomogeneous[sampleIndex] / before;
        afterHomogeneous[sampleIndex] += delta;
        double newContribution = (double) afterHomogeneous[sampleIndex] / before;
        return nsarSum - oldContribution + newContribution;
    }

    private double informationGain(double parentEntropy, int[] leftCounts, int[] rightCounts, int leftSize, int rightSize) {
        int total = leftSize + rightSize;
        double weightedEntropy = ((double) leftSize / total) * entropy(leftCounts, leftSize) + ((double) rightSize / total) * entropy(rightCounts, rightSize);
        return parentEntropy - weightedEntropy;
    }

    private double entropy(int[] counts, int total) {
        if (total == 0) return 0.0;
        double entropy = 0.0;
        for (int count : counts) {
            if (count == 0) continue;
            double p = (double) count / total;
            entropy -= p * (Math.log(p) / Math.log(2.0));
        }
        return entropy;
    }

    private static final class NsarStats {
        int count = 0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double sum = 0.0;

        void add(double value) {
            count++;
            if (value < min) min = value;
            if (value > max) max = value;
            sum += value;
        }
    }

    private int inferClassCount(List<Sample> samples) {
        int max = 0;
        for (Sample sample : samples) if (sample.getLabel() > max) max = sample.getLabel();
        return max + 1;
    }
}
