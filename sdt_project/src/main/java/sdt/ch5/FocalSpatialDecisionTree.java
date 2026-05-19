package sdt.ch5;

import sdt.model.Sample;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class FocalSpatialDecisionTree {
    private static final double TIE_EPS = 1e-12;

    private final int maxNeighborhoodSize;
    private final int minNodeSize;
    private final FocalNeighborhoodMode mode;
    private FocalNode root;

    public FocalSpatialDecisionTree(int maxNeighborhoodSize, int minNodeSize, FocalNeighborhoodMode mode) {
        if (maxNeighborhoodSize < 0) throw new IllegalArgumentException("maxNeighborhoodSize must be >= 0");
        if (minNodeSize <= 0) throw new IllegalArgumentException("minNodeSize must be > 0");
        if (mode == null) throw new IllegalArgumentException("mode must not be null");
        this.maxNeighborhoodSize = maxNeighborhoodSize;
        this.minNodeSize = minNodeSize;
        this.mode = mode;
    }

    public void fit(List<Sample> samples) {
        if (samples == null || samples.isEmpty()) throw new IllegalArgumentException("samples must not be empty");
        this.root = build(List.copyOf(samples), 0);
    }

    public int[] predictBatch(List<Sample> samples) {
        if (root == null) throw new IllegalStateException("Tree is not trained");
        int[] labels = new int[samples.size()];
        Arrays.fill(labels, -1);
        int[] all = new int[samples.size()];
        for (int i = 0; i < all.length; i++) all[i] = i;
        predictRecursive(root, samples, all, labels);
        return labels;
    }

    public double errorRateBatch(List<Sample> samples) {
        if (samples == null || samples.isEmpty()) return 0.0;
        int[] pred = predictBatch(samples);
        int err = 0;
        for (int i = 0; i < pred.length; i++) if (pred[i] != samples.get(i).getLabel()) err++;
        return (double) err / pred.length;
    }

    public FocalNode getRoot() { return root; }
    public int getMaxNeighborhoodSize() { return maxNeighborhoodSize; }
    public int getMinNodeSize() { return minNodeSize; }
    public FocalNeighborhoodMode getMode() { return mode; }

    private FocalNode build(List<Sample> samples, int depth) {
        FocalNode node = new FocalNode();
        node.setDepth(depth);
        node.setSize(samples.size());
        node.setPredictedLabel(majorityLabel(samples));
        if (samples.size() < minNodeSize || isPure(samples)) {
            node.setLeaf(true);
            return node;
        }

        BestSplit best = findBestSplitRefined(samples);
        if (best == null) {
            node.setLeaf(true);
            return node;
        }

        node.setLeaf(false);
        node.setFeatureIndex(best.featureIndex);
        node.setThreshold(best.threshold);
        node.setNeighborhoodSize(best.neighborhoodSize);
        node.setInformationGain(best.informationGain);
        node.setLeft(build(best.leftSamples, depth + 1));
        node.setRight(build(best.rightSamples, depth + 1));
        return node;
    }

    private BestSplit findBestSplitRefined(List<Sample> samples) {
        int n = samples.size();
        int featureCount = samples.get(0).getFeatureCount();
        int classCount = inferClassCount(samples);
        double parentEntropy = entropy(samples, classCount);
        BestSplit globalBest = null;

        for (int f = 0; f < featureCount; f++) {
            List<Integer> order = sortedOrder(samples, f);
            for (int s = 0; s <= maxNeighborhoodSize; s++) {
                RefinedScan scan = new RefinedScan(samples, order, f, s, classCount);
                BestSplit localBest = scan.scan(parentEntropy);
                if (isBetter(localBest, globalBest)) globalBest = localBest;
            }
        }
        return globalBest;
    }

    private boolean isBetter(BestSplit candidate, BestSplit incumbent) {
        if (candidate == null) return false;
        if (incumbent == null) return true;
        if (candidate.informationGain > incumbent.informationGain + TIE_EPS) return true;
        if (Math.abs(candidate.informationGain - incumbent.informationGain) <= TIE_EPS) {
            // Keep deterministic output and prefer smaller focal windows on ties.
            if (candidate.neighborhoodSize < incumbent.neighborhoodSize) return true;
            if (candidate.neighborhoodSize == incumbent.neighborhoodSize && candidate.featureIndex < incumbent.featureIndex) return true;
        }
        return false;
    }

    private final class RefinedScan {
        private final List<Sample> samples;
        private final List<Integer> order;
        private final int featureIndex;
        private final int s;
        private final int classCount;
        private final Map<Long, Integer> indexByCell;
        private final int[][] squareNeighbors;
        private final boolean[] localIndicator;
        private final boolean[] focalTest;
        private final int[] leftCounts;
        private final int[] rightCounts;
        private int leftSize = 0;
        private int rightSize;

        RefinedScan(List<Sample> samples, List<Integer> order, int featureIndex, int s, int classCount) {
            this.samples = samples;
            this.order = order;
            this.featureIndex = featureIndex;
            this.s = s;
            this.classCount = classCount;
            this.indexByCell = buildCellIndex(samples);
            this.squareNeighbors = buildSquareNeighbors(samples, indexByCell, s);
            this.localIndicator = new boolean[samples.size()];
            this.focalTest = new boolean[samples.size()];
            this.leftCounts = new int[classCount];
            this.rightCounts = new int[classCount];
            this.rightSize = samples.size();
            for (Sample sample : samples) rightCounts[sample.getLabel()]++;
        }

        BestSplit scan(double parentEntropy) {
            BestSplit best = null;
            int n = samples.size();
            int cursor = 0;
            while (cursor < n) {
                double threshold = samples.get(order.get(cursor)).getFeature(featureIndex);
                List<Integer> affected = new ArrayList<>();
                int movedUntil = cursor;
                while (movedUntil < n && Double.compare(samples.get(order.get(movedUntil)).getFeature(featureIndex), threshold) == 0) {
                    int idx = order.get(movedUntil);
                    if (!localIndicator[idx]) {
                        localIndicator[idx] = true;
                        addAffected(affected, idx);
                    }
                    movedUntil++;
                }
                for (int idx : affected) recomputeFocalMembership(idx);

                boolean effectiveThreshold = movedUntil < n
                        && samples.get(order.get(movedUntil - 1)).getFeature(featureIndex) < samples.get(order.get(movedUntil)).getFeature(featureIndex);
                if (effectiveThreshold && leftSize >= minNodeSize && rightSize >= minNodeSize) {
                    double ig = informationGain(parentEntropy, leftCounts, rightCounts, leftSize, rightSize);
                    if (best == null || ig > best.informationGain + TIE_EPS) {
                        SplitLists lists = materializeSplit();
                        best = new BestSplit(featureIndex, threshold, s, ig, lists.left, lists.right);
                    }
                }
                cursor = movedUntil;
            }
            return best;
        }

        private void addAffected(List<Integer> affected, int idx) {
            affected.add(idx);
            for (int nb : squareNeighbors[idx]) affected.add(nb);
        }

        private void recomputeFocalMembership(int idx) {
            boolean old = focalTest[idx];
            boolean now = focalTestValue(idx);
            if (old == now) return;
            focalTest[idx] = now;
            int label = samples.get(idx).getLabel();
            if (now) {
                leftCounts[label]++;
                rightCounts[label]--;
                leftSize++;
                rightSize--;
            } else {
                leftCounts[label]--;
                rightCounts[label]++;
                leftSize--;
                rightSize++;
            }
        }

        private SplitLists materializeSplit() {
            List<Sample> left = new ArrayList<>(leftSize);
            List<Sample> right = new ArrayList<>(rightSize);
            for (int i = 0; i < samples.size(); i++) {
                if (focalTest[i]) left.add(samples.get(i));
                else right.add(samples.get(i));
            }
            return new SplitLists(left, right);
        }

        private boolean focalTestValue(int idx) {
            if (s == 0) return localIndicator[idx];
            double gamma = (mode == FocalNeighborhoodMode.FIXED)
                    ? fixedGamma(idx, localIndicator, squareNeighbors)
                    : adaptiveGamma(idx, localIndicator, squareNeighbors, samples, indexByCell, s);
            return localIndicator[idx] ^ (gamma < 0.0);
        }
    }

    private void predictRecursive(FocalNode node, List<Sample> samples, int[] indices, int[] labels) {
        if (node.isLeaf()) {
            for (int idx : indices) labels[idx] = node.getPredictedLabel();
            return;
        }
        SplitIndexResult split = splitIndicesByFocalTest(samples, indices, node.getFeatureIndex(), node.getThreshold(), node.getNeighborhoodSize());
        predictRecursive(node.getLeft(), samples, split.left(), labels);
        predictRecursive(node.getRight(), samples, split.right(), labels);
    }

    private SplitIndexResult splitIndicesByFocalTest(List<Sample> samples, int[] indices, int featureIndex, double threshold, int s) {
        List<Sample> subset = new ArrayList<>(indices.length);
        for (int idx : indices) subset.add(samples.get(idx));
        Map<Long, Integer> indexByCell = buildCellIndex(subset);
        int[][] squareNeighbors = buildSquareNeighbors(subset, indexByCell, s);
        boolean[] local = new boolean[subset.size()];
        for (int i = 0; i < subset.size(); i++) local[i] = subset.get(i).getFeature(featureIndex) <= threshold;
        List<Integer> left = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        for (int i = 0; i < subset.size(); i++) {
            boolean value;
            if (s == 0) value = local[i];
            else {
                double gamma = (mode == FocalNeighborhoodMode.FIXED)
                        ? fixedGamma(i, local, squareNeighbors)
                        : adaptiveGamma(i, local, squareNeighbors, subset, indexByCell, s);
                value = local[i] ^ (gamma < 0.0);
            }
            if (value) left.add(indices[i]); else right.add(indices[i]);
        }
        return new SplitIndexResult(toIntArray(left), toIntArray(right));
    }

    private static double fixedGamma(int idx, boolean[] local, int[][] squareNeighbors) {
        int[] neighbors = squareNeighbors[idx];
        if (neighbors.length == 0) return 1.0;
        int iVal = local[idx] ? 1 : -1;
        double sum = 0.0;
        for (int nb : neighbors) sum += iVal * (local[nb] ? 1 : -1);
        return sum / neighbors.length;
    }

    private static double adaptiveGamma(int idx,
                                        boolean[] local,
                                        int[][] squareNeighbors,
                                        List<Sample> samples,
                                        Map<Long, Integer> indexByCell,
                                        int s) {
        int[] window = squareNeighbors[idx];
        if (window.length == 0) return 1.0;
        Sample center = samples.get(idx);
        int r0 = center.getRow();
        int c0 = center.getCol();
        int rowMin = r0 - s, rowMax = r0 + s, colMin = c0 - s, colMax = c0 + s;

        Set<Integer> windowSet = new HashSet<>();
        for (int w : window) windowSet.add(w);
        windowSet.add(idx);

        boolean[] seen = new boolean[samples.size()];
        Component centerComponent = null;
        Component largestBoundary = null;
        for (int start : windowSet) {
            if (seen[start]) continue;
            Component component = floodComponent(start, local[start], local, samples, windowSet, indexByCell, seen, rowMin, rowMax, colMin, colMax);
            if (component.members.contains(idx)) centerComponent = component;
            if (component.touchesBoundary && (largestBoundary == null || component.members.size() > largestBoundary.members.size())) {
                largestBoundary = component;
            }
        }
        Component chosen = centerComponent;
        if (centerComponent != null && !centerComponent.touchesBoundary && largestBoundary != null) {
            chosen = largestBoundary;
        }
        if (chosen == null || chosen.members.isEmpty()) return 1.0;
        int iVal = local[idx] ? 1 : -1;
        double sum = 0.0;
        int count = 0;
        for (int nb : chosen.members) {
            if (nb == idx) continue;
            sum += iVal * (local[nb] ? 1 : -1);
            count++;
        }
        return count == 0 ? 1.0 : sum / count;
    }

    private static Component floodComponent(int start,
                                            boolean value,
                                            boolean[] local,
                                            List<Sample> samples,
                                            Set<Integer> windowSet,
                                            Map<Long, Integer> indexByCell,
                                            boolean[] seen,
                                            int rowMin,
                                            int rowMax,
                                            int colMin,
                                            int colMax) {
        Component component = new Component();
        ArrayDeque<Integer> q = new ArrayDeque<>();
        q.add(start);
        seen[start] = true;
        while (!q.isEmpty()) {
            int idx = q.removeFirst();
            component.members.add(idx);
            Sample s = samples.get(idx);
            if (s.getRow() == rowMin || s.getRow() == rowMax || s.getCol() == colMin || s.getCol() == colMax) {
                component.touchesBoundary = true;
            }
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    if (dr == 0 && dc == 0) continue;
                    Integer nb = indexByCell.get(cellKey(s.getRow() + dr, s.getCol() + dc));
                    if (nb == null || seen[nb] || !windowSet.contains(nb) || local[nb] != value) continue;
                    seen[nb] = true;
                    q.add(nb);
                }
            }
        }
        return component;
    }

    private static int[][] buildSquareNeighbors(List<Sample> samples, Map<Long, Integer> indexByCell, int s) {
        int[][] result = new int[samples.size()][];
        if (s <= 0) {
            for (int i = 0; i < result.length; i++) result[i] = new int[0];
            return result;
        }
        for (int i = 0; i < samples.size(); i++) {
            Sample sample = samples.get(i);
            List<Integer> neighbors = new ArrayList<>();
            for (int dr = -s; dr <= s; dr++) {
                for (int dc = -s; dc <= s; dc++) {
                    if (dr == 0 && dc == 0) continue;
                    Integer nb = indexByCell.get(cellKey(sample.getRow() + dr, sample.getCol() + dc));
                    if (nb != null) neighbors.add(nb);
                }
            }
            result[i] = toIntArray(neighbors);
        }
        return result;
    }

    private static Map<Long, Integer> buildCellIndex(List<Sample> samples) {
        Map<Long, Integer> map = new HashMap<>(samples.size() * 2);
        for (int i = 0; i < samples.size(); i++) map.put(cellKey(samples.get(i).getRow(), samples.get(i).getCol()), i);
        return map;
    }

    private static long cellKey(int row, int col) {
        return (((long) row) << 32) ^ (col & 0xffffffffL);
    }

    private static List<Integer> sortedOrder(List<Sample> samples, int featureIndex) {
        List<Integer> order = new ArrayList<>(samples.size());
        for (int i = 0; i < samples.size(); i++) order.add(i);
        order.sort(Comparator.comparingDouble((Integer i) -> samples.get(i).getFeature(featureIndex)).thenComparingInt(i -> samples.get(i).getId()));
        return order;
    }

    private static double informationGain(double parentEntropy, int[] leftCounts, int[] rightCounts, int leftSize, int rightSize) {
        int total = leftSize + rightSize;
        return parentEntropy - ((double) leftSize / total) * entropy(leftCounts, leftSize) - ((double) rightSize / total) * entropy(rightCounts, rightSize);
    }

    private static double entropy(List<Sample> samples, int classCount) {
        int[] counts = new int[classCount];
        for (Sample s : samples) counts[s.getLabel()]++;
        return entropy(counts, samples.size());
    }

    private static double entropy(int[] counts, int total) {
        if (total == 0) return 0.0;
        double e = 0.0;
        for (int c : counts) {
            if (c == 0) continue;
            double p = (double) c / total;
            e -= p * (Math.log(p) / Math.log(2.0));
        }
        return e;
    }

    private static boolean isPure(List<Sample> samples) {
        int label = samples.get(0).getLabel();
        for (Sample s : samples) if (s.getLabel() != label) return false;
        return true;
    }

    private static int majorityLabel(List<Sample> samples) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (Sample sample : samples) counts.merge(sample.getLabel(), 1, Integer::sum);
        return counts.entrySet().stream()
                .max(Comparator.<Map.Entry<Integer, Integer>>comparingInt(Map.Entry::getValue).thenComparingInt(e -> -e.getKey()))
                .orElseThrow()
                .getKey();
    }

    private static int inferClassCount(List<Sample> samples) {
        int max = 0;
        for (Sample sample : samples) max = Math.max(max, sample.getLabel());
        return max + 1;
    }

    private static int[] toIntArray(List<Integer> list) {
        int[] out = new int[list.size()];
        for (int i = 0; i < list.size(); i++) out[i] = list.get(i);
        return out;
    }

    public static TreeStats stats(FocalNode root) {
        TreeStatsAcc acc = new TreeStatsAcc();
        stats(root, acc);
        return new TreeStats(acc.nodes, acc.leaves, acc.maxDepth, acc.minLeaf == Integer.MAX_VALUE ? 0 : acc.minLeaf, acc.maxLeaf);
    }

    private static void stats(FocalNode node, TreeStatsAcc acc) {
        if (node == null) return;
        acc.nodes++;
        acc.maxDepth = Math.max(acc.maxDepth, node.getDepth());
        if (node.isLeaf()) {
            acc.leaves++;
            acc.minLeaf = Math.min(acc.minLeaf, node.getSize());
            acc.maxLeaf = Math.max(acc.maxLeaf, node.getSize());
        } else {
            stats(node.getLeft(), acc);
            stats(node.getRight(), acc);
        }
    }

    public static String statsString(FocalNode root) {
        TreeStats s = stats(root);
        return String.format(Locale.US, "nodes=%d, leaves=%d, maxDepth=%d, minLeafSize=%d, maxLeafSize=%d",
                s.nodes(), s.leaves(), s.maxDepth(), s.minLeafSize(), s.maxLeafSize());
    }

    private record SplitLists(List<Sample> left, List<Sample> right) {}
    private record SplitIndexResult(int[] left, int[] right) {}
    private static final class Component { final List<Integer> members = new ArrayList<>(); boolean touchesBoundary; }
    private static final class TreeStatsAcc { int nodes, leaves, maxDepth, minLeaf = Integer.MAX_VALUE, maxLeaf; }
    public record TreeStats(int nodes, int leaves, int maxDepth, int minLeafSize, int maxLeafSize) {}

    private static final class BestSplit {
        final int featureIndex;
        final double threshold;
        final int neighborhoodSize;
        final double informationGain;
        final List<Sample> leftSamples;
        final List<Sample> rightSamples;
        BestSplit(int featureIndex, double threshold, int neighborhoodSize, double informationGain, List<Sample> leftSamples, List<Sample> rightSamples) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.neighborhoodSize = neighborhoodSize;
            this.informationGain = informationGain;
            this.leftSamples = leftSamples;
            this.rightSamples = rightSamples;
        }
    }
}
