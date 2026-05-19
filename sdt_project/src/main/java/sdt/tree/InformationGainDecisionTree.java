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

public final class InformationGainDecisionTree implements TreeClassifier {

    private final int minNodeSize;
    private Node root;

    public InformationGainDecisionTree(int minNodeSize) {
        if (minNodeSize <= 0) {
            throw new IllegalArgumentException("minNodeSize must be > 0");
        }
        this.minNodeSize = minNodeSize;
    }

    public void fit(List<Sample> samples) {
        if (samples == null || samples.isEmpty()) {
            throw new IllegalArgumentException("samples must not be empty");
        }
        this.root = build(List.copyOf(samples), 0);
    }

    @Override
    public int predict(double[] features) {
        if (root == null) {
            throw new IllegalStateException("Tree is not trained");
        }
        Node current = root;
        while (!current.isLeaf()) {
            if (features[current.getFeatureIndex()] <= current.getThreshold()) {
                current = current.getLeft();
            } else {
                current = current.getRight();
            }
        }
        return current.getPredictedLabel();
    }

    public void printTree(String initialIndent) {
        printTree(root, initialIndent == null ? "" : initialIndent);
    }

    public Node getRoot() {
        return root;
    }

    public int getMinNodeSize() {
        return minNodeSize;
    }

    private Node build(List<Sample> samples, int depth) {
        Node node = new Node();
        node.setDepth(depth);
        node.setSize(samples.size());
        node.setPredictedLabel(majorityLabel(samples));

        if (samples.size() < minNodeSize || isPure(samples)) {
            node.setLeaf(true);
            return node;
        }

        OrdinarySplit best = findBestSplit(samples);
        if (best == null) {
            node.setLeaf(true);
            return node;
        }

        node.setLeaf(false);
        node.setFeatureIndex(best.featureIndex());
        node.setThreshold(best.threshold());
        node.setInformationGain(best.informationGain());
        node.setNsar(Double.NaN);
        node.setSig(best.informationGain());
        node.setLeft(build(best.leftSamples(), depth + 1));
        node.setRight(build(best.rightSamples(), depth + 1));
        return node;
    }

    private OrdinarySplit findBestSplit(List<Sample> samples) {
        int n = samples.size();
        int featureCount = samples.get(0).getFeatureCount();
        int classCount = inferClassCount(samples);
        double parentEntropy = EntropyUtils.entropy(samples);

        OrdinarySplit best = null;
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            OrdinarySplit candidate = findBestSplitForFeature(samples, featureIndex, classCount, parentEntropy);
            if (candidate != null && (best == null || candidate.informationGain() > best.informationGain())) {
                best = candidate;
            }
        }
        return best;
    }

    private OrdinarySplit findBestSplitForFeature(List<Sample> samples,
                                                  int featureIndex,
                                                  int classCount,
                                                  double parentEntropy) {
        int n = samples.size();
        List<Integer> order = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            order.add(i);
        }
        order.sort(Comparator
                .comparingDouble((Integer idx) -> samples.get(idx).getFeature(featureIndex))
                .thenComparingInt(idx -> samples.get(idx).getId()));

        int[] leftCounts = new int[classCount];
        int[] rightCounts = new int[classCount];
        for (Sample sample : samples) {
            rightCounts[sample.getLabel()]++;
        }

        OrdinarySplit best = null;
        int leftSize = 0;
        int rightSize = n;
        int cursor = 0;
        while (cursor < n) {
            double threshold = samples.get(order.get(cursor)).getFeature(featureIndex);
            int movedUntil = cursor;
            while (movedUntil < n
                    && Double.compare(samples.get(order.get(movedUntil)).getFeature(featureIndex), threshold) == 0) {
                Sample moved = samples.get(order.get(movedUntil));
                leftCounts[moved.getLabel()]++;
                rightCounts[moved.getLabel()]--;
                leftSize++;
                rightSize--;
                movedUntil++;
            }

            if (leftSize >= minNodeSize && rightSize >= minNodeSize) {
                double informationGain = informationGain(parentEntropy, leftCounts, rightCounts, leftSize, rightSize);
                if (best == null || informationGain > best.informationGain()) {
                    List<Sample> leftSamples = new ArrayList<>(leftSize);
                    List<Sample> rightSamples = new ArrayList<>(rightSize);
                    for (Sample sample : samples) {
                        if (sample.getFeature(featureIndex) <= threshold) {
                            leftSamples.add(sample);
                        } else {
                            rightSamples.add(sample);
                        }
                    }
                    best = new OrdinarySplit(featureIndex, threshold, informationGain, leftSamples, rightSamples);
                }
            }
            cursor = movedUntil;
        }
        return best;
    }

    private static double informationGain(double parentEntropy,
                                          int[] leftCounts,
                                          int[] rightCounts,
                                          int leftSize,
                                          int rightSize) {
        int total = leftSize + rightSize;
        double weightedEntropy = ((double) leftSize / total) * entropy(leftCounts, leftSize)
                + ((double) rightSize / total) * entropy(rightCounts, rightSize);
        return parentEntropy - weightedEntropy;
    }

    private static double entropy(int[] counts, int total) {
        if (total == 0) {
            return 0.0;
        }
        double entropy = 0.0;
        for (int count : counts) {
            if (count == 0) {
                continue;
            }
            double p = (double) count / total;
            entropy -= p * (Math.log(p) / Math.log(2.0));
        }
        return entropy;
    }

    private static boolean isPure(List<Sample> samples) {
        int firstLabel = samples.get(0).getLabel();
        for (Sample sample : samples) {
            if (sample.getLabel() != firstLabel) {
                return false;
            }
        }
        return true;
    }

    private static int majorityLabel(List<Sample> samples) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (Sample sample : samples) {
            counts.merge(sample.getLabel(), 1, Integer::sum);
        }
        return counts.entrySet().stream()
                .max(Comparator.<Map.Entry<Integer, Integer>>comparingInt(Map.Entry::getValue)
                        .thenComparingInt(entry -> -entry.getKey()))
                .orElseThrow()
                .getKey();
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
            System.out.println(indent + "Leaf(label=" + node.getPredictedLabel() + ", size=" + node.getSize() + ")");
        } else {
            System.out.println(indent + "Node(f[" + node.getFeatureIndex() + "] <= " + node.getThreshold()
                    + ", size=" + node.getSize()
                    + ", depth=" + node.getDepth()
                    + ", IG=" + String.format(Locale.US, "%.4f", node.getInformationGain()) + ")");
            printTree(node.getLeft(), indent + "  YES -> ");
            printTree(node.getRight(), indent + "  NO  -> ");
        }
    }

    private record OrdinarySplit(int featureIndex,
                                 double threshold,
                                 double informationGain,
                                 List<Sample> leftSamples,
                                 List<Sample> rightSamples) {
    }
}
