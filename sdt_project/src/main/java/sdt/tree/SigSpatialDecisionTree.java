package sdt.tree;

import sdt.model.Node;
import sdt.model.Sample;
import sdt.model.SplitResult;
import sdt.spatial.NeighborhoodDefinition;
import sdt.spatial.SpatialGraph;
import sdt.spatial.SpatialGraphBuilder;
import sdt.split.SplitFinder;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public final class SigSpatialDecisionTree implements TreeClassifier {

    private final double alpha;
    private final int minNodeSize;
    private final NeighborhoodDefinition neighborhoodDefinition;
    private final SplitFinder splitFinder;
    private Node root;

    public SigSpatialDecisionTree(double alpha, int minNodeSize) {
        this(alpha, minNodeSize, NeighborhoodDefinition.rook());
    }

    
    public SigSpatialDecisionTree(double alpha, int minNodeSize, int ignoredMaxDepth) {
        this(alpha, minNodeSize, NeighborhoodDefinition.rook());
    }

    public SigSpatialDecisionTree(double alpha, int minNodeSize, NeighborhoodDefinition neighborhoodDefinition) {
        if (alpha < 0.0 || alpha > 1.0) {
            throw new IllegalArgumentException("alpha must be in [0, 1]");
        }
        if (minNodeSize <= 0) {
            throw new IllegalArgumentException("minNodeSize must be > 0");
        }
        if (neighborhoodDefinition == null) {
            throw new IllegalArgumentException("neighborhoodDefinition must not be null");
        }
        this.alpha = alpha;
        this.minNodeSize = minNodeSize;
        this.neighborhoodDefinition = neighborhoodDefinition;
        this.splitFinder = new SplitFinder(alpha, minNodeSize);
    }

    public void fit(List<Sample> samples) {
        SpatialGraph graph = SpatialGraphBuilder.build(samples, neighborhoodDefinition);
        fit(graph);
    }

    public void fit(SpatialGraph graph) {
        if (graph == null || graph.size() == 0) {
            throw new IllegalArgumentException("graph is empty");
        }
        this.root = build(graph, 0);
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

    public double errorRate(List<Sample> samples) {
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

    public void printTree() {
        printTree(root, "");
    }

    public void printTree(String initialIndent) {
        printTree(root, initialIndent == null ? "" : initialIndent);
    }

    public Node getRoot() {
        return root;
    }

    public double getAlpha() {
        return alpha;
    }

    public int getMinNodeSize() {
        return minNodeSize;
    }

    public NeighborhoodDefinition getNeighborhoodDefinition() {
        return neighborhoodDefinition;
    }

    public boolean isTrained() {
        return root != null;
    }

    private Node build(SpatialGraph graph, int depth) {
        Node node = new Node();
        node.setDepth(depth);
        node.setSize(graph.size());
        node.setPredictedLabel(majorityLabel(graph.samples()));

        if (graph.size() < minNodeSize || isPure(graph.samples())) {
            node.setLeaf(true);
            return node;
        }

        SplitResult bestSplit = splitFinder.findBestSplit(graph);
        if (bestSplit == null) {
            node.setLeaf(true);
            return node;
        }

        node.setLeaf(false);
        node.setFeatureIndex(bestSplit.getFeatureIndex());
        node.setThreshold(bestSplit.getThreshold());
        node.setInformationGain(bestSplit.getInformationGain());
        node.setNsar(bestSplit.getNsar());
        node.setSig(bestSplit.getSig());

        node.setLeft(build(bestSplit.getLeftGraph(), depth + 1));
        node.setRight(build(bestSplit.getRightGraph(), depth + 1));
        return node;
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
                    + ", IG=" + String.format("%.4f", node.getInformationGain())
                    + ", NSAR=" + String.format("%.4f", node.getNsar())
                    + ", SIG=" + String.format("%.4f", node.getSig()) + ")");
            printTree(node.getLeft(), indent + "  YES -> ");
            printTree(node.getRight(), indent + "  NO  -> ");
        }
    }

    private boolean isPure(List<Sample> samples) {
        if (samples.isEmpty()) {
            return true;
        }
        int firstLabel = samples.get(0).getLabel();
        for (Sample sample : samples) {
            if (sample.getLabel() != firstLabel) {
                return false;
            }
        }
        return true;
    }

    private int majorityLabel(List<Sample> samples) {
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
}
