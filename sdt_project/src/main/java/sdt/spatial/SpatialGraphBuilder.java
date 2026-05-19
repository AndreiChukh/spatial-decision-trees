package sdt.spatial;

import sdt.model.Sample;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


public final class SpatialGraphBuilder {

    private SpatialGraphBuilder() {
    }


    public static SpatialGraph buildNaive(Collection<Sample> samples, NeighborhoodDefinition definition) {
        if (samples == null || samples.isEmpty()) {
            throw new IllegalArgumentException("samples must not be empty");
        }
        if (definition == null) {
            throw new IllegalArgumentException("definition must not be null");
        }

        Map<Integer, Sample> samplesById = new LinkedHashMap<>();
        for (Sample sample : samples) {
            if (samplesById.put(sample.getId(), sample) != null) {
                throw new IllegalArgumentException("duplicate sample id: " + sample.getId());
            }
        }

        Map<Integer, Set<Integer>> adjacency = new LinkedHashMap<>();
        for (Sample sample : samplesById.values()) {
            adjacency.put(sample.getId(), new LinkedHashSet<>());
        }

        buildPairwiseGraph(samplesById.values().toArray(new Sample[0]), adjacency, definition);
        return new SpatialGraph(samplesById, adjacency);
    }

    public static SpatialGraph build(Collection<Sample> samples, NeighborhoodDefinition definition) {
        if (samples == null || samples.isEmpty()) {
            throw new IllegalArgumentException("samples must not be empty");
        }
        if (definition == null) {
            throw new IllegalArgumentException("definition must not be null");
        }

        Map<Integer, Sample> samplesById = new LinkedHashMap<>();
        for (Sample sample : samples) {
            if (samplesById.put(sample.getId(), sample) != null) {
                throw new IllegalArgumentException("duplicate sample id: " + sample.getId());
            }
        }

        Map<Integer, Set<Integer>> adjacency = new LinkedHashMap<>();
        for (Sample sample : samplesById.values()) {
            adjacency.put(sample.getId(), new LinkedHashSet<>());
        }

        if (definition.getType() == NeighborhoodType.DISTANCE_THRESHOLD) {
            buildDistanceThresholdGraph(samplesById.values(), adjacency, definition.getDistanceThreshold());
        } else {
            buildPairwiseGraph(samplesById.values().toArray(new Sample[0]), adjacency, definition);
        }

        return new SpatialGraph(samplesById, adjacency);
    }

    private static void buildPairwiseGraph(Sample[] array,
                                           Map<Integer, Set<Integer>> adjacency,
                                           NeighborhoodDefinition definition) {
        for (int i = 0; i < array.length; i++) {
            for (int j = i + 1; j < array.length; j++) {
                if (areNeighbors(array[i], array[j], definition)) {
                    addUndirectedEdge(adjacency, array[i], array[j]);
                }
            }
        }
    }

    
    private static void buildDistanceThresholdGraph(Collection<Sample> samples,
                                                    Map<Integer, Set<Integer>> adjacency,
                                                    double radius) {
        int cellSize = Math.max(1, (int) Math.ceil(radius));
        double radiusSquared = radius * radius;
        Map<Long, List<Sample>> buckets = new LinkedHashMap<>();

        for (Sample sample : samples) {
            int bucketRow = Math.floorDiv(sample.getRow(), cellSize);
            int bucketCol = Math.floorDiv(sample.getCol(), cellSize);
            long key = bucketKey(bucketRow, bucketCol);

            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    List<Sample> candidates = buckets.get(bucketKey(bucketRow + dr, bucketCol + dc));
                    if (candidates == null) {
                        continue;
                    }
                    for (Sample other : candidates) {
                        if (squaredDistance(sample, other) <= radiusSquared) {
                            addUndirectedEdge(adjacency, sample, other);
                        }
                    }
                }
            }

            buckets.computeIfAbsent(key, ignored -> new ArrayList<>()).add(sample);
        }
    }

    private static long bucketKey(int row, int col) {
        return (((long) row) << 32) ^ (col & 0xffffffffL);
    }

    private static void addUndirectedEdge(Map<Integer, Set<Integer>> adjacency, Sample a, Sample b) {
        adjacency.get(a.getId()).add(b.getId());
        adjacency.get(b.getId()).add(a.getId());
    }

    private static boolean areNeighbors(Sample a, Sample b, NeighborhoodDefinition definition) {
        int dr = Math.abs(a.getRow() - b.getRow());
        int dc = Math.abs(a.getCol() - b.getCol());

        return switch (definition.getType()) {
            case ROOK -> (dr == 1 && dc == 0) || (dr == 0 && dc == 1);
            case QUEEN -> (dr <= 1 && dc <= 1) && (dr + dc > 0);
            case DISTANCE_THRESHOLD -> squaredDistance(a, b) <= definition.getDistanceThreshold() * definition.getDistanceThreshold();
        };
    }

    private static double squaredDistance(Sample a, Sample b) {
        int dr = a.getRow() - b.getRow();
        int dc = a.getCol() - b.getCol();
        return (double) dr * dr + (double) dc * dc;
    }
}
