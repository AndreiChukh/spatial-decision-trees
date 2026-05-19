package sdt.spatial;

import sdt.model.Sample;

import java.util.List;


public final class SpatialUtils {

    private SpatialUtils() {
    }

    public static SpatialGraph buildRookGraph(List<Sample> samples) {
        return SpatialGraphBuilder.build(samples, NeighborhoodDefinition.rook());
    }

    public static SpatialGraph buildQueenGraph(List<Sample> samples) {
        return SpatialGraphBuilder.build(samples, NeighborhoodDefinition.queen());
    }

    public static SpatialGraph buildDistanceThresholdGraph(List<Sample> samples, double distanceThreshold) {
        return SpatialGraphBuilder.build(samples, NeighborhoodDefinition.distanceThreshold(distanceThreshold));
    }
}
