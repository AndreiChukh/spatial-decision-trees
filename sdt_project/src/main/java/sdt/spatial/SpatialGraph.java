package sdt.spatial;

import sdt.model.Sample;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


public final class SpatialGraph {

    private final Map<Integer, Sample> samplesById;
    private final Map<Integer, Set<Integer>> adjacency;

    public SpatialGraph(Map<Integer, Sample> samplesById, Map<Integer, Set<Integer>> adjacency) {
        this.samplesById = new LinkedHashMap<>(samplesById);
        this.adjacency = new LinkedHashMap<>();
        for (Map.Entry<Integer, Set<Integer>> entry : adjacency.entrySet()) {
            this.adjacency.put(entry.getKey(), new LinkedHashSet<>(entry.getValue()));
        }
    }

    public List<Sample> samples() {
        return List.copyOf(samplesById.values());
    }

    public int size() {
        return samplesById.size();
    }

    public boolean contains(int sampleId) {
        return samplesById.containsKey(sampleId);
    }

    public Sample getSample(int sampleId) {
        return samplesById.get(sampleId);
    }

    public List<Sample> neighborsOf(int sampleId) {
        Set<Integer> neighborIds = adjacency.getOrDefault(sampleId, Collections.emptySet());
        List<Sample> neighbors = new ArrayList<>(neighborIds.size());
        for (int neighborId : neighborIds) {
            Sample neighbor = samplesById.get(neighborId);
            if (neighbor != null) {
                neighbors.add(neighbor);
            }
        }
        return neighbors;
    }

    public Set<Integer> neighborIdsOf(int sampleId) {
        return Collections.unmodifiableSet(adjacency.getOrDefault(sampleId, Collections.emptySet()));
    }

    public SpatialGraph inducedSubgraph(Collection<Sample> subset) {
        Map<Integer, Sample> subsetSamples = new LinkedHashMap<>();
        for (Sample sample : subset) {
            subsetSamples.put(sample.getId(), sample);
        }

        Map<Integer, Set<Integer>> subsetAdjacency = new LinkedHashMap<>();
        for (Integer sampleId : subsetSamples.keySet()) {
            Set<Integer> sourceNeighbors = adjacency.getOrDefault(sampleId, Collections.emptySet());
            Set<Integer> filtered = new LinkedHashSet<>();
            for (Integer neighborId : sourceNeighbors) {
                if (subsetSamples.containsKey(neighborId)) {
                    filtered.add(neighborId);
                }
            }
            subsetAdjacency.put(sampleId, filtered);
        }

        return new SpatialGraph(subsetSamples, subsetAdjacency);
    }
}
