package sdt.spatial;


public final class NeighborhoodDefinition {

    private final NeighborhoodType type;
    private final double distanceThreshold;

    private NeighborhoodDefinition(NeighborhoodType type, double distanceThreshold) {
        this.type = type;
        this.distanceThreshold = distanceThreshold;
    }

    public static NeighborhoodDefinition rook() {
        return new NeighborhoodDefinition(NeighborhoodType.ROOK, 1.0);
    }

    public static NeighborhoodDefinition queen() {
        return new NeighborhoodDefinition(NeighborhoodType.QUEEN, Math.sqrt(2.0));
    }

    public static NeighborhoodDefinition distanceThreshold(double distanceThreshold) {
        if (distanceThreshold <= 0.0) {
            throw new IllegalArgumentException("distanceThreshold must be > 0");
        }
        return new NeighborhoodDefinition(NeighborhoodType.DISTANCE_THRESHOLD, distanceThreshold);
    }

    public NeighborhoodType getType() {
        return type;
    }

    public double getDistanceThreshold() {
        return distanceThreshold;
    }
}
