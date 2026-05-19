package sdt.tree;

import sdt.metrics.EntropyUtils;
import sdt.model.Node;
import sdt.model.Sample;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class C45DecisionTree implements TreeClassifier {
    private static final double EPS = 1e-12;
    private static final double DEFAULT_CONFIDENCE_FACTOR = 0.25;

    private final int minNodeSize;
    private final double confidenceFactor;
    private final boolean pruningEnabled;
    private final boolean subtreeRaisingEnabled;
    private final boolean mdlContinuousCorrectionEnabled;

    private Node root;
    private int classCount;

    public C45DecisionTree(int minNodeSize) {
        this(minNodeSize, DEFAULT_CONFIDENCE_FACTOR, true, true, true);
    }

    public C45DecisionTree(int minNodeSize,
                           double confidenceFactor,
                           boolean pruningEnabled,
                           boolean subtreeRaisingEnabled,
                           boolean mdlContinuousCorrectionEnabled) {
        if (minNodeSize <= 0) {
            throw new IllegalArgumentException("minNodeSize must be > 0");
        }
        if (confidenceFactor <= 0.0 || confidenceFactor >= 0.5) {
            throw new IllegalArgumentException("confidenceFactor must be in (0, 0.5)");
        }
        this.minNodeSize = minNodeSize;
        this.confidenceFactor = confidenceFactor;
        this.pruningEnabled = pruningEnabled;
        this.subtreeRaisingEnabled = subtreeRaisingEnabled;
        this.mdlContinuousCorrectionEnabled = mdlContinuousCorrectionEnabled;
    }

    public void fit(List<Sample> samples) {
        if (samples == null || samples.isEmpty()) {
            throw new IllegalArgumentException("samples must not be empty");
        }
        this.classCount = inferClassCount(samples);
        List<WeightedSample> weighted = new ArrayList<>(samples.size());
        for (Sample sample : samples) {
            weighted.add(new WeightedSample(sample, 1.0));
        }
        this.root = build(weighted, 0);
        if (pruningEnabled) {
            prune(root, weighted);
        }
    }

    @Override
    public int predict(double[] features) {
        if (root == null) {
            throw new IllegalStateException("Tree is not trained");
        }
        double[] distribution = predictDistribution(root, features);
        return argMax(distribution);
    }

    public Node getRoot() {
        return root;
    }

    public int getMinNodeSize() {
        return minNodeSize;
    }

    public double getConfidenceFactor() {
        return confidenceFactor;
    }

    public boolean isPruningEnabled() {
        return pruningEnabled;
    }

    public void printTree(String initialIndent) {
        printTree(root, initialIndent == null ? "" : initialIndent);
    }

    private Node build(List<WeightedSample> samples, int depth) {
        Node node = new Node();
        node.setDepth(depth);
        node.setSize((int) Math.round(totalWeight(samples)));
        double[] classWeights = classWeights(samples);
        node.setClassWeights(classWeights);
        node.setPredictedLabel(argMax(classWeights));

        if (totalWeight(samples) < minNodeSize || isPure(classWeights)) {
            node.setLeaf(true);
            return node;
        }

        C45Split best = findBestSplit(samples);
        if (best == null || best.gainRatio() <= EPS || best.informationGain() <= EPS) {
            node.setLeaf(true);
            return node;
        }

        SplitBuckets buckets = splitSamples(samples, best);
        if (totalWeight(buckets.left()) < EPS || totalWeight(buckets.right()) < EPS) {
            node.setLeaf(true);
            return node;
        }

        node.setLeaf(false);
        node.setFeatureIndex(best.featureIndex());
        node.setThreshold(best.threshold());
        node.setInformationGain(best.informationGain());
        node.setNsar(Double.NaN);
        node.setSig(best.gainRatio());
        node.setBranchWeights(new double[]{totalWeight(buckets.left()), totalWeight(buckets.right())});
        node.setLeft(build(buckets.left(), depth + 1));
        node.setRight(build(buckets.right(), depth + 1));
        return node;
    }

    private C45Split findBestSplit(List<WeightedSample> samples) {
        int featureCount = samples.get(0).sample().getFeatureCount();
        double parentEntropy = entropy(classWeights(samples));

        List<C45Split> candidates = new ArrayList<>();
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            C45Split candidate = findBestSplitForFeature(samples, featureIndex, parentEntropy);
            if (candidate != null && candidate.informationGain() > EPS && candidate.gainRatio() > EPS) {
                candidates.add(candidate);
            }
        }
        if (candidates.isEmpty()) {
            return null;
        }

        double averageGain = 0.0;
        for (C45Split split : candidates) {
            averageGain += split.informationGain();
        }
        averageGain /= candidates.size();

        C45Split best = null;
        for (C45Split candidate : candidates) {
            if (candidate.informationGain() + EPS < averageGain) {
                continue;
            }
            if (isBetter(candidate, best)) {
                best = candidate;
            }
        }
        if (best != null) {
            return best;
        }

        for (C45Split candidate : candidates) {
            if (isBetter(candidate, best)) {
                best = candidate;
            }
        }
        return best;
    }

    private boolean isBetter(C45Split candidate, C45Split best) {
        if (best == null) {
            return true;
        }
        if (candidate.gainRatio() > best.gainRatio() + EPS) {
            return true;
        }
        if (Math.abs(candidate.gainRatio() - best.gainRatio()) <= EPS) {
            if (candidate.informationGain() > best.informationGain() + EPS) {
                return true;
            }
            if (Math.abs(candidate.informationGain() - best.informationGain()) <= EPS) {
                return candidate.balance() > best.balance() + EPS;
            }
        }
        return false;
    }

    private C45Split findBestSplitForFeature(List<WeightedSample> samples,
                                             int featureIndex,
                                             double parentEntropy) {
        List<WeightedSample> known = new ArrayList<>();
        double unknownWeight = 0.0;
        for (WeightedSample sample : samples) {
            double value = sample.sample().getFeature(featureIndex);
            if (Double.isNaN(value)) {
                unknownWeight += sample.weight();
            } else {
                known.add(sample);
            }
        }
        if (known.size() < 2 || totalWeight(known) < minNodeSize) {
            return null;
        }

        known.sort(Comparator
                .comparingDouble((WeightedSample ws) -> ws.sample().getFeature(featureIndex))
                .thenComparingInt(ws -> ws.sample().getId()));

        List<Integer> candidatePositions = new ArrayList<>();
        for (int i = 0; i < known.size() - 1; i++) {
            double current = known.get(i).sample().getFeature(featureIndex);
            double next = known.get(i + 1).sample().getFeature(featureIndex);
            if (Double.compare(current, next) == 0) {
                continue;
            }
            if (known.get(i).sample().getLabel() != known.get(i + 1).sample().getLabel()) {
                candidatePositions.add(i);
            }
        }
        if (candidatePositions.isEmpty()) {
            return null;
        }

        double[] leftCounts = new double[classCount];
        double[] rightCounts = classWeights(known);
        double leftKnownWeight = 0.0;
        double rightKnownWeight = totalWeight(known);
        double totalWeight = leftKnownWeight + rightKnownWeight + unknownWeight;
        double knownWeight = rightKnownWeight;
        double knownFraction = knownWeight / totalWeight;
        int nextCandidateIndex = 0;

        C45Split best = null;
        for (int i = 0; i < known.size() - 1; i++) {
            WeightedSample moved = known.get(i);
            int label = moved.sample().getLabel();
            leftCounts[label] += moved.weight();
            rightCounts[label] -= moved.weight();
            leftKnownWeight += moved.weight();
            rightKnownWeight -= moved.weight();

            if (nextCandidateIndex >= candidatePositions.size() || candidatePositions.get(nextCandidateIndex) != i) {
                continue;
            }
            nextCandidateIndex++;

            if (leftKnownWeight < minNodeSize || rightKnownWeight < minNodeSize) {
                continue;
            }

            double currentValue = known.get(i).sample().getFeature(featureIndex);
            double nextValue = known.get(i + 1).sample().getFeature(featureIndex);
            double threshold = midpoint(currentValue, nextValue);
            double knownEntropyAfter = (leftKnownWeight / knownWeight) * entropy(leftCounts)
                    + (rightKnownWeight / knownWeight) * entropy(rightCounts);
            double informationGain = knownFraction * (parentEntropy - knownEntropyAfter);

            if (mdlContinuousCorrectionEnabled && candidatePositions.size() > 1) {
                informationGain -= log2(candidatePositions.size()) / totalWeight;
            }

            double splitInfo = splitInfo(leftKnownWeight, rightKnownWeight, unknownWeight, totalWeight);
            if (splitInfo <= EPS || informationGain <= EPS) {
                continue;
            }
            double gainRatio = informationGain / splitInfo;
            double balance = Math.min(leftKnownWeight, rightKnownWeight) / knownWeight;
            C45Split candidate = new C45Split(featureIndex, threshold, informationGain, splitInfo,
                    gainRatio, balance, leftKnownWeight, rightKnownWeight);
            if (isBetter(candidate, best)) {
                best = candidate;
            }
        }
        return best;
    }

    private SplitBuckets splitSamples(List<WeightedSample> samples, C45Split split) {
        List<WeightedSample> left = new ArrayList<>();
        List<WeightedSample> right = new ArrayList<>();
        double knownWeight = split.leftKnownWeight() + split.rightKnownWeight();
        double leftFraction = knownWeight <= EPS ? 0.5 : split.leftKnownWeight() / knownWeight;
        double rightFraction = 1.0 - leftFraction;
        for (WeightedSample ws : samples) {
            double value = ws.sample().getFeature(split.featureIndex());
            if (Double.isNaN(value)) {
                if (leftFraction > EPS) {
                    left.add(new WeightedSample(ws.sample(), ws.weight() * leftFraction));
                }
                if (rightFraction > EPS) {
                    right.add(new WeightedSample(ws.sample(), ws.weight() * rightFraction));
                }
            } else if (value <= split.threshold()) {
                left.add(ws);
            } else {
                right.add(ws);
            }
        }
        return new SplitBuckets(left, right);
    }

    private PruneResult prune(Node node, List<WeightedSample> samples) {
        if (node == null) {
            return new PruneResult(0.0, 0.0);
        }

        double total = totalWeight(samples);
        double[] weights = classWeights(samples);
        node.setClassWeights(weights);
        node.setSize((int) Math.round(total));
        node.setPredictedLabel(argMax(weights));

        if (node.isLeaf()) {
            double errors = leafErrors(weights);
            return new PruneResult(errors, estimatedErrors(errors, total));
        }

        C45Split split = new C45Split(node.getFeatureIndex(), node.getThreshold(),
                node.getInformationGain(), Double.NaN, node.getSig(), 0.0,
                node.getBranchWeights()[0], node.getBranchWeights()[1]);
        SplitBuckets buckets = splitSamples(samples, split);
        PruneResult leftResult = prune(node.getLeft(), buckets.left());
        PruneResult rightResult = prune(node.getRight(), buckets.right());

        double subtreeEstimatedErrors = leftResult.estimatedErrors() + rightResult.estimatedErrors();
        double leafErrors = leafErrors(weights);
        double leafEstimatedErrors = estimatedErrors(leafErrors, total);

        Node raised = null;
        double raisedEstimatedErrors = Double.POSITIVE_INFINITY;
        if (subtreeRaisingEnabled) {
            Node left = node.getLeft();
            Node right = node.getRight();
            double leftRaised = estimatedSubtreeErrorsOnSamples(left, samples);
            double rightRaised = estimatedSubtreeErrorsOnSamples(right, samples);
            if (leftRaised < rightRaised) {
                raised = left;
                raisedEstimatedErrors = leftRaised;
            } else {
                raised = right;
                raisedEstimatedErrors = rightRaised;
            }
        }

        if (leafEstimatedErrors <= subtreeEstimatedErrors + 0.1
                && leafEstimatedErrors <= raisedEstimatedErrors + 0.1) {
            makeLeaf(node, weights, total);
            return new PruneResult(leafErrors, leafEstimatedErrors);
        }

        if (subtreeRaisingEnabled && raised != null && raisedEstimatedErrors + 0.1 < subtreeEstimatedErrors) {
            copyNodeInto(raised, node);
            resetDepths(node, node.getDepth());
            double rawErrors = rawErrors(node, samples);
            return new PruneResult(rawErrors, raisedEstimatedErrors);
        }

        double rawErrors = rawErrors(node, samples);
        return new PruneResult(rawErrors, subtreeEstimatedErrors);
    }

    private void makeLeaf(Node node, double[] weights, double total) {
        node.setLeaf(true);
        node.setPredictedLabel(argMax(weights));
        node.setClassWeights(weights);
        node.setSize((int) Math.round(total));
        node.setLeft(null);
        node.setRight(null);
        node.setInformationGain(0.0);
        node.setNsar(Double.NaN);
        node.setSig(0.0);
        node.setBranchWeights(null);
    }

    private void copyNodeInto(Node source, Node target) {
        target.setLeaf(source.isLeaf());
        target.setPredictedLabel(source.getPredictedLabel());
        target.setFeatureIndex(source.getFeatureIndex());
        target.setThreshold(source.getThreshold());
        target.setLeft(source.getLeft());
        target.setRight(source.getRight());
        target.setSize(source.getSize());
        target.setInformationGain(source.getInformationGain());
        target.setNsar(source.getNsar());
        target.setSig(source.getSig());
        target.setClassWeights(source.getClassWeights());
        target.setBranchWeights(source.getBranchWeights());
    }

    private void resetDepths(Node node, int depth) {
        if (node == null) {
            return;
        }
        node.setDepth(depth);
        resetDepths(node.getLeft(), depth + 1);
        resetDepths(node.getRight(), depth + 1);
    }

    private double estimatedSubtreeErrorsOnSamples(Node subtree, List<WeightedSample> samples) {
        double errors = rawErrors(subtree, samples);
        return estimatedErrors(errors, totalWeight(samples));
    }

    private double rawErrors(Node node, List<WeightedSample> samples) {
        double errors = 0.0;
        for (WeightedSample sample : samples) {
            double[] distribution = predictDistribution(node, sample.sample().getFeatures());
            int predicted = argMax(distribution);
            if (predicted != sample.sample().getLabel()) {
                errors += sample.weight();
            }
        }
        return errors;
    }

    private double[] predictDistribution(Node node, double[] features) {
        if (node == null) {
            double[] empty = new double[classCount];
            if (empty.length > 0) {
                empty[0] = 1.0;
            }
            return empty;
        }
        if (node.isLeaf()) {
            return normalizedDistribution(node.getClassWeights(), node.getPredictedLabel());
        }
        double value = features[node.getFeatureIndex()];
        if (Double.isNaN(value)) {
            double[] branchWeights = node.getBranchWeights();
            if (branchWeights == null || branchWeights.length < 2) {
                return normalizedDistribution(node.getClassWeights(), node.getPredictedLabel());
            }
            double total = branchWeights[0] + branchWeights[1];
            if (total <= EPS) {
                return normalizedDistribution(node.getClassWeights(), node.getPredictedLabel());
            }
            double[] left = predictDistribution(node.getLeft(), features);
            double[] right = predictDistribution(node.getRight(), features);
            double[] merged = new double[classCount];
            double leftFraction = branchWeights[0] / total;
            double rightFraction = branchWeights[1] / total;
            for (int i = 0; i < classCount; i++) {
                merged[i] = leftFraction * left[i] + rightFraction * right[i];
            }
            return merged;
        }
        if (value <= node.getThreshold()) {
            return predictDistribution(node.getLeft(), features);
        }
        return predictDistribution(node.getRight(), features);
    }

    private double[] normalizedDistribution(double[] weights, int fallbackLabel) {
        double[] result = new double[classCount];
        if (weights != null) {
            double total = 0.0;
            for (int i = 0; i < Math.min(weights.length, classCount); i++) {
                result[i] = Math.max(0.0, weights[i]);
                total += result[i];
            }
            if (total > EPS) {
                for (int i = 0; i < result.length; i++) {
                    result[i] /= total;
                }
                return result;
            }
        }
        if (fallbackLabel >= 0 && fallbackLabel < classCount) {
            result[fallbackLabel] = 1.0;
        } else if (classCount > 0) {
            result[0] = 1.0;
        }
        return result;
    }

    private double estimatedErrors(double errors, double total, double confidence) {
        if (total <= EPS) {
            return 0.0;
        }
        double z = inverseNormal(1.0 - confidence);
        double p = Math.max(0.0, Math.min(errors, total)) / total;
        double z2 = z * z;
        double denominator = 1.0 + z2 / total;
        double center = p + z2 / (2.0 * total);
        double margin = z * Math.sqrt((p * (1.0 - p) / total) + z2 / (4.0 * total * total));
        return total * ((center + margin) / denominator);
    }

    private double estimatedErrors(double errors, double total) {
        return estimatedErrors(errors, total, confidenceFactor);
    }

    private static double leafErrors(double[] classWeights) {
        double total = 0.0;
        double max = 0.0;
        for (double weight : classWeights) {
            total += weight;
            if (weight > max) {
                max = weight;
            }
        }
        return total - max;
    }

    private static double entropy(double[] classWeights) {
        double total = 0.0;
        for (double weight : classWeights) {
            total += weight;
        }
        if (total <= EPS) {
            return 0.0;
        }
        double result = 0.0;
        for (double weight : classWeights) {
            if (weight <= EPS) {
                continue;
            }
            double p = weight / total;
            result -= p * log2(p);
        }
        return result;
    }

    private static double splitInfo(double leftKnownWeight,
                                    double rightKnownWeight,
                                    double unknownWeight,
                                    double totalWeight) {
        double result = 0.0;
        result -= splitInfoTerm(leftKnownWeight / totalWeight);
        result -= splitInfoTerm(rightKnownWeight / totalWeight);
        if (unknownWeight > EPS) {
            result -= splitInfoTerm(unknownWeight / totalWeight);
        }
        return result;
    }

    private static double splitInfoTerm(double p) {
        return p <= EPS ? 0.0 : p * log2(p);
    }

    private static boolean isPure(double[] classWeights) {
        int positive = 0;
        for (double weight : classWeights) {
            if (weight > EPS) {
                positive++;
            }
        }
        return positive <= 1;
    }

    private double[] classWeights(List<WeightedSample> samples) {
        double[] weights = new double[classCount];
        for (WeightedSample sample : samples) {
            weights[sample.sample().getLabel()] += sample.weight();
        }
        return weights;
    }

    private static double totalWeight(List<WeightedSample> samples) {
        double total = 0.0;
        for (WeightedSample sample : samples) {
            total += sample.weight();
        }
        return total;
    }

    private static double midpoint(double a, double b) {
        return a + (b - a) / 2.0;
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }

    private static int argMax(double[] values) {
        int best = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > bestValue + EPS) {
                best = i;
                bestValue = values[i];
            }
        }
        return best;
    }

    private static int inferClassCount(List<Sample> samples) {
        int max = 0;
        for (Sample sample : samples) {
            if (sample.getLabel() > max) {
                max = sample.getLabel();
            }
        }
        return max + 1;
    }

    private void printTree(Node node, String indent) {
        if (node == null) {
            return;
        }
        if (node.isLeaf()) {
            System.out.println(indent + "Leaf(label=" + node.getPredictedLabel()
                    + ", size=" + node.getSize() + ")");
        } else {
            System.out.println(indent + "Node(f[" + node.getFeatureIndex() + "] <= " + node.getThreshold()
                    + ", size=" + node.getSize()
                    + ", depth=" + node.getDepth()
                    + ", IG=" + String.format(Locale.US, "%.4f", node.getInformationGain())
                    + ", GainRatio=" + String.format(Locale.US, "%.4f", node.getSig()) + ")");
            printTree(node.getLeft(), indent + "  YES -> ");
            printTree(node.getRight(), indent + "  NO  -> ");
        }
    }

    private static double inverseNormal(double p) {
        if (p <= 0.0 || p >= 1.0) {
            throw new IllegalArgumentException("p must be in (0,1)");
        }
        double[] a = {
                -3.969683028665376e+01,
                2.209460984245205e+02,
                -2.759285104469687e+02,
                1.383577518672690e+02,
                -3.066479806614716e+01,
                2.506628277459239e+00
        };
        double[] b = {
                -5.447609879822406e+01,
                1.615858368580409e+02,
                -1.556989798598866e+02,
                6.680131188771972e+01,
                -1.328068155288572e+01
        };
        double[] c = {
                -7.784894002430293e-03,
                -3.223964580411365e-01,
                -2.400758277161838e+00,
                -2.549732539343734e+00,
                4.374664141464968e+00,
                2.938163982698783e+00
        };
        double[] d = {
                7.784695709041462e-03,
                3.224671290700398e-01,
                2.445134137142996e+00,
                3.754408661907416e+00
        };
        double plow = 0.02425;
        double phigh = 1.0 - plow;
        if (p < plow) {
            double q = Math.sqrt(-2.0 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
        if (p > phigh) {
            double q = Math.sqrt(-2.0 * Math.log(1.0 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }

    private record WeightedSample(Sample sample, double weight) {
    }

    private record SplitBuckets(List<WeightedSample> left, List<WeightedSample> right) {
    }

    private record C45Split(int featureIndex,
                            double threshold,
                            double informationGain,
                            double splitInfo,
                            double gainRatio,
                            double balance,
                            double leftKnownWeight,
                            double rightKnownWeight) {
    }

    private record PruneResult(double rawErrors, double estimatedErrors) {
    }
}
