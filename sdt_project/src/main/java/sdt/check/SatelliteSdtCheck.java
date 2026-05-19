package sdt.check;

import sdt.io.RasterStatistics;
import sdt.io.SingleBandRaster;
import sdt.io.TiffReader;
import sdt.model.Sample;
import sdt.model.Node;
import sdt.spatial.NeighborhoodDefinition;
import sdt.spatial.SpatialGraph;
import sdt.spatial.SpatialGraphBuilder;
import sdt.tree.C45DecisionTree;
import sdt.tree.SigSpatialDecisionTree;
import sdt.tree.TreeClassifier;
import sdt.tuning.AlphaTuner;
import sdt.viz.ChartUtils;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

public final class SatelliteSdtCheck {
    private static final int DRY = 0;
    private static final int WET = 1;
    private static final int CLASS_COUNT = 2;

    private static final int DEFAULT_NODATA = 0;
    private static final int DEFAULT_PSU_SIZE = 98;
    private static final int DEFAULT_SAMPLES_PER_PSU = 1000;
    private static final int DEFAULT_TRAIN_PSU_PER_CLASS = 8;
    private static final int DEFAULT_VALIDATION_PSU_PER_CLASS = 2;
    private static final double DEFAULT_TRAINING_AREA_FRACTION = 0.70;
    private static final int DEFAULT_TEST_EVAL_STEP = 1;

    private static final double DEFAULT_ALPHA_START = 0.02;
    private static final double DEFAULT_ALPHA_END = 1.00;
    private static final double DEFAULT_ALPHA_STEP = 0.02;
    private static final int DEFAULT_MIN_NODE_SIZE = 100;
    private static final double DEFAULT_NEIGHBOR_DISTANCE = 6.0;
    private static final long DEFAULT_RANDOM_SEED = 42L;
    private static final double DEFAULT_NDVI_WET_THRESHOLD = 0.10;
    private static final int DEFAULT_PERFORMANCE_SAMPLE_LIMIT = 5000;
    private static final int DEFAULT_TREE_PRINT_LEAF_LIMIT = 64;
    private static final Path DEFAULT_OUTPUT_DIR = Path.of("outputs_1");
    private static final int CLEAR_QA_PIXEL = 21824;

    private SatelliteSdtCheck() {
    }

    public static void run(String[] args) throws IOException {
        if (args.length < 2) {
            printUsage();
            return;
        }

        int requiredPathCount = detectRequiredPathCount(args);
        ParsedRunArguments parsed = parseRunArguments(args, requiredPathCount);
        List<String> positional = parsed.positionalExtras();

        int psuSize = positional.size() >= 1 ? Integer.parseInt(positional.get(0)) : DEFAULT_PSU_SIZE;
        int samplesPerPsu = positional.size() >= 2 ? Integer.parseInt(positional.get(1)) : DEFAULT_SAMPLES_PER_PSU;
        int trainPsuPerClass = positional.size() >= 3 ? Integer.parseInt(positional.get(2)) : DEFAULT_TRAIN_PSU_PER_CLASS;
        int validationPsuPerClass = positional.size() >= 4 ? Integer.parseInt(positional.get(3)) : DEFAULT_VALIDATION_PSU_PER_CLASS;
        double trainingAreaFraction = parsed.trainValidationFraction() != null
                ? parsed.trainValidationFraction()
                : (positional.size() >= 5 ? Double.parseDouble(positional.get(4)) : DEFAULT_TRAINING_AREA_FRACTION);
        int testEvalStep = positional.size() >= 6 ? Integer.parseInt(positional.get(5)) : DEFAULT_TEST_EVAL_STEP;
        int noData = positional.size() >= 7 ? Integer.parseInt(positional.get(6)) : DEFAULT_NODATA;
        double alphaStart = positional.size() >= 8 ? Double.parseDouble(positional.get(7)) : DEFAULT_ALPHA_START;
        double alphaEnd = positional.size() >= 9 ? Double.parseDouble(positional.get(8)) : DEFAULT_ALPHA_END;
        // Apply --alpha-end override if supplied
        for (String extra : parsed.positionalExtras()) {
            if (extra.startsWith("__alphaEnd=")) {
                alphaEnd = Double.parseDouble(extra.substring("__alphaEnd=".length()));
            }
        }
        double alphaStep = positional.size() >= 10 ? Double.parseDouble(positional.get(9)) : DEFAULT_ALPHA_STEP;
        int minNodeSize = parsed.minNodeSizeOption() != null
                ? parsed.minNodeSizeOption()
                : (positional.size() >= 11 ? Integer.parseInt(positional.get(10)) : DEFAULT_MIN_NODE_SIZE);
        double neighborDistance = positional.size() >= 12 ? Double.parseDouble(positional.get(11)) : DEFAULT_NEIGHBOR_DISTANCE;
        long randomSeed = positional.size() >= 13 ? Long.parseLong(positional.get(12)) : DEFAULT_RANDOM_SEED;
        double ndviWetThreshold = parsed.ndviWetThreshold() != null
                ? parsed.ndviWetThreshold()
                : (positional.size() >= 14 ? Double.parseDouble(positional.get(13)) : DEFAULT_NDVI_WET_THRESHOLD);
        int performanceSampleLimit = parsed.performanceSampleLimit() != null
                ? parsed.performanceSampleLimit()
                : DEFAULT_PERFORMANCE_SAMPLE_LIMIT;
        Path outputDir = parsed.outputDir() != null ? parsed.outputDir() : DEFAULT_OUTPUT_DIR;
        int treePrintLeafLimit = parsed.treePrintLeafLimit() != null
                ? parsed.treePrintLeafLimit()
                : DEFAULT_TREE_PRINT_LEAF_LIMIT;
        Double fixedAlpha = parsed.fixedAlpha();

        validateParameters(psuSize, samplesPerPsu, trainPsuPerClass, validationPsuPerClass,
                trainingAreaFraction, testEvalStep, alphaStart, alphaEnd, alphaStep,
                minNodeSize, neighborDistance, ndviWetThreshold, performanceSampleLimit);
        if (fixedAlpha != null && (fixedAlpha < 0.0 || fixedAlpha > 1.0)) {
            throw new IllegalArgumentException("--alpha must be in [0, 1]");
        }
        Files.createDirectories(outputDir);

        RasterBundle rasters = readRasterBundle(args, requiredPathCount, noData);
        printRasterStats(rasters);

        int trainingEndRow = Math.max(1, Math.min(rasters.height() - 1,
                (int) Math.round(rasters.height() * trainingAreaFraction)));
        Area trainingArea = new Area(0, trainingEndRow, 0, rasters.width());
        Area testArea = new Area(trainingEndRow, rasters.height(), 0, rasters.width());
        NeighborhoodDefinition neighborhood = NeighborhoodDefinition.distanceThreshold(neighborDistance);

        System.out.println("\nTrain/validation/test evaluation with binary NDVI pseudo-labels (low-NDVI vs high-NDVI):");
        System.out.println("  ground truth note: low-NDVI / high-NDVI pseudo-labels are generated from NDVI");
        System.out.println("  training/validation area: top rows [0, " + trainingEndRow + ")"
                + " = " + percent(trainingAreaFraction));
        System.out.println("  independent test area   : bottom rows [" + trainingEndRow + ", " + rasters.height() + ")"
                + " = " + percent(1.0 - trainingAreaFraction));
        System.out.println("  PSU size=" + psuSize + "x" + psuSize + " pixels");
        System.out.println("  train PSUs/class=" + trainPsuPerClass
                + ", validation PSUs/class=" + validationPsuPerClass
                + ", samples/PSU=" + samplesPerPsu);
        System.out.println("  test area evaluation step=" + testEvalStep + " pixel(s)");
        System.out.println("  neighborDistance=" + neighborDistance
                + " px, minNodeSize=" + minNodeSize
                + " (minimum child/leaf size guard)");
        System.out.println("  spatial split: top " + percent(trainingAreaFraction)
                + " of rows used for train/validation; bottom " + percent(1.0 - trainingAreaFraction)
                + " reserved for test");
        System.out.println("  PSU selection order: random shuffle within each majority class");
        System.out.println("  pixel sampling seed=" + randomSeed);
        System.out.println("  outputDir=" + outputDir.toAbsolutePath());
        printLabelLegend(ndviWetThreshold);

        BookStyleSplit split = buildBookStyleSplit(rasters, trainingArea, psuSize,
                trainPsuPerClass, validationPsuPerClass, samplesPerPsu, randomSeed, ndviWetThreshold);

        System.out.println("\nSelected PSUs by majority NDVI class:");
        System.out.println("  train PSU counts     : " + formatClassArray(split.trainPsuCounts()));
        System.out.println("  validation PSU counts: " + formatClassArray(split.validationPsuCounts()));
        System.out.println("\nSamples:");
        System.out.println("  Train size     : " + split.trainSamples().size() + ", labels " + countLabels(split.trainSamples()));
        System.out.println("  Validation size: " + split.validationSamples().size() + ", labels " + countLabels(split.validationSamples()));

        validateSamples(split.trainSamples(), split.validationSamples(), minNodeSize);

        List<PerformanceRow> performanceRows = new ArrayList<>();
        performanceRows.addAll(measureGraphBuilders(split.trainSamples(), neighborhood, performanceSampleLimit));

        TimedResult<C45DecisionTree> dtTimed = time("C4.5-DT train", () -> {
            C45DecisionTree tree = new C45DecisionTree(minNodeSize);
            tree.fit(split.trainSamples());
            return tree;
        });
        performanceRows.add(new PerformanceRow("C4.5-DT train", split.trainSamples().size(), 0,
                dtTimed.millis(), "C4.5 tree: gain ratio, missing-value fractions, pessimistic pruning"));
        C45DecisionTree dt = dtTimed.value();

        double selectedAlpha;
        AlphaTuner.TuningResult tuning = null;
        if (fixedAlpha != null) {
            selectedAlpha = fixedAlpha;
            System.out.println("\nAlpha selection:");
            System.out.printf(Locale.US, "  using fixed alpha from command line: %.4f%n", selectedAlpha);
            System.out.println("  validation alpha tuning is skipped");
        } else {
            List<Double> alphas = candidateAlphas(alphaStart, alphaEnd, alphaStep);
            System.out.println("\nAlpha tuning on validation set:");
            System.out.println("  alpha candidates: " + alphas.size()
                    + " values from " + alphaStart + " to " + alphaEnd + " step " + alphaStep);

            tuning = AlphaTuner.tune(
                    split.trainSamples(),
                    split.validationSamples(),
                    alphas,
                    minNodeSize,
                    neighborhood
            );

            for (AlphaTuner.EvaluationPoint point : tuning.evaluations()) {
                System.out.printf(Locale.US, "  alpha=%.2f trainError=%.4f validationError=%.4f%n",
                        point.alpha(), point.trainingError(), point.validationError());
            }
            selectedAlpha = tuning.alpha();
            System.out.printf(Locale.US, "Best alpha = %.2f, validation error = %.4f%n",
                    selectedAlpha, tuning.validationError());
            writeAlphaTuningCsv(outputDir.resolve("alpha_tuning.csv"), tuning.evaluations());
        }

        TimedResult<SigSpatialDecisionTree> sdtTimed = time("SDT train", () -> {
            SigSpatialDecisionTree tree = new SigSpatialDecisionTree(selectedAlpha, minNodeSize, neighborhood);
            tree.fit(split.trainSamples());
            return tree;
        });
        performanceRows.add(new PerformanceRow("SDT train", split.trainSamples().size(), 0,
                sdtTimed.millis(), "SIG tree with optimized spatial graph"));
        SigSpatialDecisionTree sdt = sdtTimed.value();

        System.out.println("\nSelected C4.5-DT baseline:");
        System.out.printf(Locale.US, "  Train error     : %.4f%n", dt.errorRate(split.trainSamples()));
        System.out.printf(Locale.US, "  Validation error: %.4f%n", dt.errorRate(split.validationSamples()));
        printTreeStats("C4.5-DT", computeTreeStats(dt.getRoot()));
        maybePrintTree(dt.getRoot(), "    ", treePrintLeafLimit);

        System.out.println("\nSelected SDT:");
        System.out.printf(Locale.US, "  alpha           : %.4f%n", selectedAlpha);
        System.out.printf(Locale.US, "  Train error     : %.4f%n", sdt.errorRate(split.trainSamples()));
        System.out.printf(Locale.US, "  Validation error: %.4f%n", sdt.errorRate(split.validationSamples()));
        printTreeStats("SDT", computeTreeStats(sdt.getRoot()));
        maybePrintTree(sdt.getRoot(), "    ", treePrintLeafLimit);

        TimedResult<StreamingEvaluation> dtEvalTimed = time("C4.5-DT test evaluation", () ->
                evaluateArea(dt, rasters, testArea, testEvalStep, ndviWetThreshold));
        performanceRows.add(new PerformanceRow("C4.5-DT test evaluation", (int) Math.min(Integer.MAX_VALUE, dtEvalTimed.value().validPixels()), 0,
                dtEvalTimed.millis(), "single pass over test area"));
        StreamingEvaluation dtEvaluation = dtEvalTimed.value();

        TimedResult<StreamingEvaluation> sdtEvalTimed = time("SDT test evaluation", () ->
                evaluateArea(sdt, rasters, testArea, testEvalStep, ndviWetThreshold));
        performanceRows.add(new PerformanceRow("SDT test evaluation", (int) Math.min(Integer.MAX_VALUE, sdtEvalTimed.value().validPixels()), 0,
                sdtEvalTimed.millis(), "single pass over test area"));
        StreamingEvaluation sdtEvaluation = sdtEvalTimed.value();

        BbJoinCount dtBb = bbJoinCount(dtEvaluation.predictedLabels(), dtEvaluation.mapWidth(), dtEvaluation.mapHeight(), WET);
        BbJoinCount sdtBb = bbJoinCount(sdtEvaluation.predictedLabels(), sdtEvaluation.mapWidth(), sdtEvaluation.mapHeight(), WET);

        System.out.println("\nConfusion matrix of C4.5-DT:");
        printBookConfusionMatrix(dtEvaluation.confusionMatrix());
        System.out.printf(Locale.US, "  C4.5-DT test error: %.6f%n", dtEvaluation.errorRate());

        System.out.println("\nConfusion matrix of SDT:");
        printBookConfusionMatrix(sdtEvaluation.confusionMatrix());
        System.out.printf(Locale.US, "  SDT test error: %.6f%n", sdtEvaluation.errorRate());

        System.out.println("\nBB Join Count on predicted high-NDVI class:");
        printBbJoinCountTable(dtBb, sdtBb);

        System.out.println("\nSummary:");
        System.out.println("  accuracy improved? " + yesNo(sdtEvaluation.errorRate() < dtEvaluation.errorRate())
                + " (C4.5-DT error=" + fmt6(dtEvaluation.errorRate()) + ", SDT error=" + fmt6(sdtEvaluation.errorRate()) + ")");
        System.out.println("  salt-and-pepper reduced? " + yesNo(sdtBb.normalized() > dtBb.normalized())
                + " (higher normalized BB join count = stronger spatial autocorrelation of high-NDVI class)"
                + " C4.5-DT=" + fmt4(dtBb.normalized()) + ", SDT=" + fmt4(sdtBb.normalized()) + ")");
        System.out.println("  alpha chosen by validation? " + (fixedAlpha == null ? "YES, alpha=" + fmt4(selectedAlpha) : "NO, fixed alpha=" + fmt4(selectedAlpha)));

        writeConfusionCsv(outputDir.resolve("table_c45_dt_confusion.csv"), dtEvaluation.confusionMatrix());
        writeConfusionCsv(outputDir.resolve("table_sdt_confusion.csv"), sdtEvaluation.confusionMatrix());
        writeBbCsv(outputDir.resolve("table_bb_join_count.csv"), dtBb, sdtBb);
        writePerformanceCsv(outputDir.resolve("performance_table.csv"), performanceRows);
        writeMaps(outputDir, dtEvaluation, sdtEvaluation);
        writeRgbVisualizations(outputDir, rasters);
        writeCharts(outputDir, performanceRows, tuning, dtEvaluation, sdtEvaluation, dtBb, sdtBb);

        System.out.println("\nPerformance table:");
        printPerformanceTable(performanceRows);
        System.out.println("\nFiles written:");
        System.out.println("  " + outputDir.resolve("table_c45_dt_confusion.csv").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("table_sdt_confusion.csv").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("table_bb_join_count.csv").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("performance_table.csv").toAbsolutePath());
        if (tuning != null) {
            System.out.println("  " + outputDir.resolve("alpha_tuning.csv").toAbsolutePath());
            System.out.println("  " + outputDir.resolve("alpha_tuning_chart.png").toAbsolutePath());
        }
        System.out.println("  " + outputDir.resolve("performance_chart.png").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("c45_sdt_error_chart.png").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("bb_join_chart.png").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("fig_truth.png").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("fig_c45_dt.png").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("fig_sdt.png").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("fig_combined.png").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("true_color_rgb_b4_b3_b2.png").toAbsolutePath());
        System.out.println("  " + outputDir.resolve("false_color_b5_b4_b3.png").toAbsolutePath());
    }


    private static TreeStats computeTreeStats(Node root) {
        TreeStatsAccumulator acc = new TreeStatsAccumulator();
        accumulateTreeStats(root, acc);
        int minLeafSize = acc.leaves == 0 ? 0 : acc.minLeafSize;
        return new TreeStats(acc.nodes, acc.leaves, acc.maxDepth, minLeafSize, acc.maxLeafSize);
    }

    private static void accumulateTreeStats(Node node, TreeStatsAccumulator acc) {
        if (node == null) {
            return;
        }
        acc.nodes++;
        acc.maxDepth = Math.max(acc.maxDepth, node.getDepth());
        if (node.isLeaf()) {
            acc.leaves++;
            acc.minLeafSize = Math.min(acc.minLeafSize, node.getSize());
            acc.maxLeafSize = Math.max(acc.maxLeafSize, node.getSize());
            return;
        }
        accumulateTreeStats(node.getLeft(), acc);
        accumulateTreeStats(node.getRight(), acc);
    }

    private static void printTreeStats(String name, TreeStats stats) {
        System.out.printf(Locale.US,
                "  %s tree stats : nodes=%d, leaves=%d, maxDepth=%d, minLeafSize=%d, maxLeafSize=%d%n",
                name, stats.nodes(), stats.leaves(), stats.maxDepth(), stats.minLeafSize(), stats.maxLeafSize());
    }

    private static void maybePrintTree(Node root, String indent, int treePrintLeafLimit) {
        TreeStats stats = computeTreeStats(root);
        if (stats.leaves() <= treePrintLeafLimit) {
            System.out.println("  Tree:");
            printTree(root, indent == null ? "" : indent);
        } else {
            System.out.println("  Tree: " + stats.leaves() + " leaves (limit=" + treePrintLeafLimit + ") — too large to print; see stats above");
        }
    }

    private static void printTree(Node node, String indent) {
        if (node == null) {
            return;
        }
        if (node.isLeaf()) {
            System.out.println(indent + "Leaf(label=" + node.getPredictedLabel() + ", size=" + node.getSize() + ")");
        } else {
            if (Double.isNaN(node.getNsar())) {
                System.out.println(indent + "Node(f[" + node.getFeatureIndex() + "] <= " + node.getThreshold()
                        + ", size=" + node.getSize()
                        + ", depth=" + node.getDepth()
                        + ", IG=" + String.format(Locale.US, "%.4f", node.getInformationGain())
                        + ", GainRatio=" + String.format(Locale.US, "%.4f", node.getSig()) + ")");
            } else {
                System.out.println(indent + "Node(f[" + node.getFeatureIndex() + "] <= " + node.getThreshold()
                        + ", size=" + node.getSize()
                        + ", depth=" + node.getDepth()
                        + ", IG=" + String.format(Locale.US, "%.4f", node.getInformationGain())
                        + ", NSAR=" + String.format(Locale.US, "%.4f", node.getNsar())
                        + ", SIG=" + String.format(Locale.US, "%.4f", node.getSig()) + ")");
            }
            printTree(node.getLeft(), indent + "  YES -> ");
            printTree(node.getRight(), indent + "  NO  -> ");
        }
    }

    private static BookStyleSplit buildBookStyleSplit(RasterBundle rasters,
                                                      Area area,
                                                      int psuSize,
                                                      int trainPsuPerClass,
                                                      int validationPsuPerClass,
                                                      int samplesPerPsu,
                                                      long randomSeed,
                                                      double ndviWetThreshold) {
        List<PsuInfo> psus = scanPsus(rasters, area, psuSize, ndviWetThreshold);
        Map<Integer, List<PsuInfo>> byMajorityClass = new HashMap<>();
        for (int label = 0; label < CLASS_COUNT; label++) {
            byMajorityClass.put(label, new ArrayList<>());
        }
        for (PsuInfo psu : psus) {
            byMajorityClass.get(psu.majorityClass()).add(psu);
        }

        List<Sample> train = new ArrayList<>();
        List<Sample> validation = new ArrayList<>();
        int[] trainPsuCounts = new int[CLASS_COUNT];
        int[] validationPsuCounts = new int[CLASS_COUNT];
        int nextId = 0;
        Random random = new Random(randomSeed);

        for (int label = 0; label < CLASS_COUNT; label++) {
            List<PsuInfo> eligible = new ArrayList<>(byMajorityClass.get(label));
            Collections.shuffle(eligible, random);

            int required = trainPsuPerClass + validationPsuPerClass;
            int selected = Math.min(required, eligible.size());
            if (selected < required) {
                System.out.println("WARNING: only " + selected + " majority-" + labelName(label)
                        + " PSUs available; requested " + required);
            }
            int trainSelected = Math.min(trainPsuPerClass, selected);
            int validationSelected = Math.min(validationPsuPerClass, selected - trainSelected);

            for (int i = 0; i < trainSelected; i++) {
                List<Sample> samples = samplePixelsFromPsu(rasters, eligible.get(i), samplesPerPsu,
                        nextId, random, ndviWetThreshold);
                nextId += samples.size();
                train.addAll(samples);
                trainPsuCounts[label]++;
            }
            for (int i = trainSelected; i < trainSelected + validationSelected; i++) {
                List<Sample> samples = samplePixelsFromPsu(rasters, eligible.get(i), samplesPerPsu,
                        nextId, random, ndviWetThreshold);
                nextId += samples.size();
                validation.addAll(samples);
                validationPsuCounts[label]++;
            }
        }

        return new BookStyleSplit(train, validation, trainPsuCounts, validationPsuCounts);
    }

    private static List<PsuInfo> scanPsus(RasterBundle rasters,
                                          Area area,
                                          int psuSize,
                                          double ndviWetThreshold) {
        List<PsuInfo> psus = new ArrayList<>();
        for (int rowStart = area.rowStart(); rowStart < area.rowEnd(); rowStart += psuSize) {
            int rowEnd = Math.min(rowStart + psuSize, area.rowEnd());
            for (int colStart = area.colStart(); colStart < area.colEnd(); colStart += psuSize) {
                int colEnd = Math.min(colStart + psuSize, area.colEnd());
                int[] counts = new int[CLASS_COUNT];
                int valid = 0;
                for (int row = rowStart; row < rowEnd; row++) {
                    for (int col = colStart; col < colEnd; col++) {
                        if (!isValid(rasters, row, col)) {
                            continue;
                        }
                        counts[ndviClass(ndvi(rasters.b4().get(row, col), rasters.b5().get(row, col)), ndviWetThreshold)]++;
                        valid++;
                    }
                }
                if (valid > 0) {
                    int majorityClass = counts[WET] > counts[DRY] ? WET : DRY;
                    psus.add(new PsuInfo(rowStart, rowEnd, colStart, colEnd, counts, majorityClass));
                }
            }
        }
        return psus;
    }

    private static List<Sample> samplePixelsFromPsu(RasterBundle rasters,
                                                    PsuInfo psu,
                                                    int samplesPerPsu,
                                                    int firstId,
                                                    Random random,
                                                    double ndviWetThreshold) {
        List<PixelRef> candidates = collectValidPixels(rasters, psu);
        List<Sample> samples = new ArrayList<>();
        if (candidates.isEmpty()) {
            return samples;
        }

        Collections.shuffle(candidates, random);
        int id = firstId;
        int limit = Math.min(samplesPerPsu, candidates.size());
        for (int i = 0; i < limit; i++) {
            PixelRef pixel = candidates.get(i);
            samples.add(toSample(rasters, pixel.row(), pixel.col(), id++, ndviWetThreshold));
        }
        return samples;
    }

    private static List<PixelRef> collectValidPixels(RasterBundle rasters, PsuInfo psu) {
        List<PixelRef> pixels = new ArrayList<>();
        for (int row = psu.rowStart(); row < psu.rowEnd(); row++) {
            for (int col = psu.colStart(); col < psu.colEnd(); col++) {
                if (!isValid(rasters, row, col)) {
                    continue;
                }
                pixels.add(new PixelRef(row, col));
            }
        }
        return pixels;
    }

    private static Sample toSample(RasterBundle rasters,
                                   int row,
                                   int col,
                                   int id,
                                   double ndviWetThreshold) {
        int red = rasters.b4().get(row, col);
        int nir = rasters.b5().get(row, col);
        double ndvi = ndvi(red, nir);
        double[] features = new double[]{
                rasters.b2().get(row, col),
                rasters.b3().get(row, col),
                rasters.b6().get(row, col),
                rasters.b7().get(row, col)
        };
        return new Sample(id, row, col, features, ndviClass(ndvi, ndviWetThreshold));
    }

    private static StreamingEvaluation evaluateArea(TreeClassifier tree,
                                                    RasterBundle rasters,
                                                    Area area,
                                                    int step,
                                                    double ndviWetThreshold) {
        int mapHeight = ceilDiv(area.rowEnd() - area.rowStart(), step);
        int mapWidth = ceilDiv(area.colEnd() - area.colStart(), step);
        int[] truthLabels = filledIntArray(mapHeight * mapWidth, -1);
        int[] predictedLabels = filledIntArray(mapHeight * mapWidth, -1);
        long[] predictedCounts = new long[CLASS_COUNT];
        long[] truthCounts = new long[CLASS_COUNT];
        long[][] confusion = new long[CLASS_COUNT][CLASS_COUNT];
        long valid = 0;
        long skipped = 0;
        long errors = 0;

        int mapRow = 0;
        for (int row = area.rowStart(); row < area.rowEnd(); row += step, mapRow++) {
            int mapCol = 0;
            for (int col = area.colStart(); col < area.colEnd(); col += step, mapCol++) {
                int mapIndex = mapRow * mapWidth + mapCol;
                if (!isValid(rasters, row, col)) {
                    skipped++;
                    continue;
                }
                int red = rasters.b4().get(row, col);
                int nir = rasters.b5().get(row, col);
                int truth = ndviClass(ndvi(red, nir), ndviWetThreshold);
                int predicted = tree.predict(new double[]{
                        rasters.b2().get(row, col),
                        rasters.b3().get(row, col),
                        rasters.b6().get(row, col),
                        rasters.b7().get(row, col)
                });
                valid++;
                truthLabels[mapIndex] = truth;
                predictedLabels[mapIndex] = predicted;
                truthCounts[truth]++;
                if (predicted >= 0 && predicted < CLASS_COUNT) {
                    predictedCounts[predicted]++;
                    confusion[truth][predicted]++;
                }
                if (predicted != truth) {
                    errors++;
                }
            }
        }
        return new StreamingEvaluation(mapWidth, mapHeight, valid, skipped, errors,
                predictedCounts, truthCounts, confusion, truthLabels, predictedLabels);
    }

    private static List<PerformanceRow> measureGraphBuilders(List<Sample> trainSamples,
                                                             NeighborhoodDefinition neighborhood,
                                                             int performanceSampleLimit) {
        List<PerformanceRow> rows = new ArrayList<>();
        int limit = Math.min(performanceSampleLimit, trainSamples.size());
        List<Sample> subset = new ArrayList<>(trainSamples.subList(0, limit));
        if (subset.isEmpty()) {
            return rows;
        }

        TimedResult<SpatialGraph> optimized = time("optimized graph build", () ->
                SpatialGraphBuilder.build(subset, neighborhood));
        int optimizedEdges = countUndirectedEdges(optimized.value());
        rows.add(new PerformanceRow("optimized distance graph build", subset.size(), optimizedEdges,
                optimized.millis(), "grid bucket spatial index"));

        TimedResult<SpatialGraph> naive = time("naive graph build", () ->
                SpatialGraphBuilder.buildNaive(subset, neighborhood));
        int naiveEdges = countUndirectedEdges(naive.value());
        rows.add(new PerformanceRow("naive distance graph build", subset.size(), naiveEdges,
                naive.millis(), "pairwise O(n^2) check"));

        return rows;
    }

    private static int countUndirectedEdges(SpatialGraph graph) {
        int directedLinks = 0;
        for (Sample sample : graph.samples()) {
            directedLinks += graph.neighborIdsOf(sample.getId()).size();
        }
        return directedLinks / 2;
    }

    private static BbJoinCount bbJoinCount(int[] labels, int width, int height, int wetLabel) {
        long validNodes = 0;
        long wetNodes = 0;
        long edges = 0;
        long observed = 0;
        long sharedEdgePairs = 0;
        int[] degrees = new int[labels.length];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int index = row * width + col;
                if (labels[index] < 0) {
                    continue;
                }
                validNodes++;
                if (labels[index] == wetLabel) {
                    wetNodes++;
                }
                if (col + 1 < width && labels[index + 1] >= 0) {
                    edges++;
                    degrees[index]++;
                    degrees[index + 1]++;
                    if (labels[index] == wetLabel && labels[index + 1] == wetLabel) {
                        observed++;
                    }
                }
                if (row + 1 < height && labels[index + width] >= 0) {
                    edges++;
                    degrees[index]++;
                    degrees[index + width]++;
                    if (labels[index] == wetLabel && labels[index + width] == wetLabel) {
                        observed++;
                    }
                }
            }
        }

        for (int degree : degrees) {
            if (degree >= 2) {
                sharedEdgePairs += ((long) degree * (degree - 1)) / 2;
            }
        }

        double expected = 0.0;
        double variance = 0.0;
        if (validNodes >= 2 && wetNodes >= 2 && edges > 0) {
            double p2 = fallingRatio(wetNodes, validNodes, 2);
            double p3 = fallingRatio(wetNodes, validNodes, 3);
            double p4 = fallingRatio(wetNodes, validNodes, 4);
            expected = edges * p2;
            double edgePairs = ((double) edges * (edges - 1)) / 2.0;
            double disjointEdgePairs = Math.max(0.0, edgePairs - sharedEdgePairs);
            variance = edges * (p2 - p2 * p2)
                    + 2.0 * sharedEdgePairs * (p3 - p2 * p2)
                    + 2.0 * disjointEdgePairs * (p4 - p2 * p2);
            if (variance < 0.0 && variance > -1e-6) {
                variance = 0.0;
            }
        }
        double normalized = variance > 0.0 ? (observed - expected) / Math.sqrt(variance) : 0.0;
        return new BbJoinCount(observed, expected, variance, normalized, validNodes, wetNodes, edges);
    }

    private static double fallingRatio(long numeratorCount, long denominatorCount, int order) {
        if (numeratorCount < order || denominatorCount < order) {
            return 0.0;
        }
        double num = 1.0;
        double den = 1.0;
        for (int i = 0; i < order; i++) {
            num *= (numeratorCount - i);
            den *= (denominatorCount - i);
        }
        return num / den;
    }

    private static void writeCharts(Path outputDir,
                                    List<PerformanceRow> performanceRows,
                                    AlphaTuner.TuningResult tuning,
                                    StreamingEvaluation dtEvaluation,
                                    StreamingEvaluation sdtEvaluation,
                                    BbJoinCount dtBb,
                                    BbJoinCount sdtBb) throws IOException {
        String[] perfLabels = new String[performanceRows.size()];
        double[] perfValues = new double[performanceRows.size()];
        for (int i = 0; i < performanceRows.size(); i++) {
            perfLabels[i] = performanceRows.get(i).operation();
            perfValues[i] = performanceRows.get(i).millis();
        }
        ChartUtils.writeBarChart(outputDir.resolve("performance_chart.png"),
                "Performance", "milliseconds", perfLabels, perfValues);

        ChartUtils.writeBarChart(outputDir.resolve("c45_sdt_error_chart.png"),
                "Test Error: C4.5-DT vs SDT", "error rate",
                new String[]{"C4.5-DT", "SDT"},
                new double[]{dtEvaluation.errorRate(), sdtEvaluation.errorRate()});

        ChartUtils.writeBarChart(outputDir.resolve("bb_join_chart.png"),
                "BB Join Count Normalized Score", "(JC-mu)/sigma",
                new String[]{"C4.5-DT", "SDT"},
                new double[]{dtBb.normalized(), sdtBb.normalized()});

        if (tuning != null) {
            List<AlphaTuner.EvaluationPoint> points = tuning.evaluations();
            double[] alpha = new double[points.size()];
            double[] train = new double[points.size()];
            double[] validation = new double[points.size()];
            for (int i = 0; i < points.size(); i++) {
                alpha[i] = points.get(i).alpha();
                train[i] = points.get(i).trainingError();
                validation[i] = points.get(i).validationError();
            }
            ChartUtils.writeLineChart(outputDir.resolve("alpha_tuning_chart.png"),
                    "Alpha Tuning", "error rate",
                    new String[]{"train", "validation"},
                    alpha, new double[][]{train, validation});
        }
    }

    private static void writeMaps(Path outputDir,
                                  StreamingEvaluation dtEvaluation,
                                  StreamingEvaluation sdtEvaluation) throws IOException {
        BufferedImage truth = makeTruthImage(dtEvaluation.truthLabels(), dtEvaluation.mapWidth(), dtEvaluation.mapHeight());
        BufferedImage dt = makeConfusionImage(dtEvaluation.truthLabels(), dtEvaluation.predictedLabels(), dtEvaluation.mapWidth(), dtEvaluation.mapHeight());
        BufferedImage sdt = makeConfusionImage(sdtEvaluation.truthLabels(), sdtEvaluation.predictedLabels(), sdtEvaluation.mapWidth(), sdtEvaluation.mapHeight());
        ImageIO.write(truth, "png", outputDir.resolve("fig_truth.png").toFile());
        ImageIO.write(dt, "png", outputDir.resolve("fig_c45_dt.png").toFile());
        ImageIO.write(sdt, "png", outputDir.resolve("fig_sdt.png").toFile());
        ImageIO.write(combineFigure(truth, dt, sdt), "png", outputDir.resolve("fig_combined.png").toFile());
    }

    private static BufferedImage makeTruthImage(int[] truthLabels, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < truthLabels.length; i++) {
            int color = switch (truthLabels[i]) {
                case WET -> Color.GREEN.getRGB();
                case DRY -> Color.RED.getRGB();
                default -> Color.DARK_GRAY.getRGB();
            };
            image.setRGB(i % width, i / width, color);
        }
        return image;
    }

    private static BufferedImage makeConfusionImage(int[] truthLabels, int[] predictedLabels, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < truthLabels.length; i++) {
            int truth = truthLabels[i];
            int predicted = predictedLabels[i];
            int color;
            if (truth < 0 || predicted < 0) {
                color = Color.DARK_GRAY.getRGB();
            } else if (truth == WET && predicted == WET) {
                color = Color.GREEN.getRGB();
            } else if (truth == DRY && predicted == DRY) {
                color = Color.RED.getRGB();
            } else if (truth == DRY && predicted == WET) {
                color = Color.BLUE.getRGB();
            } else {
                color = Color.BLACK.getRGB();
            }
            image.setRGB(i % width, i / width, color);
        }
        return image;
    }

    private static BufferedImage combineFigure(BufferedImage truth, BufferedImage dt, BufferedImage sdt) {
        int labelHeight = 28;
        int gap = 10;
        int width = truth.getWidth() + dt.getWidth() + sdt.getWidth() + 2 * gap;
        int height = Math.max(truth.getHeight(), Math.max(dt.getHeight(), sdt.getHeight())) + labelHeight;
        BufferedImage combined = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = combined.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 14));
        int x = 0;
        g.drawString("(a) Truth", x + 6, 18);
        g.drawImage(truth, x, labelHeight, null);
        x += truth.getWidth() + gap;
        g.drawString("(b) C4.5-DT", x + 6, 18);
        g.drawImage(dt, x, labelHeight, null);
        x += dt.getWidth() + gap;
        g.drawString("(c) SDT", x + 6, 18);
        g.drawImage(sdt, x, labelHeight, null);
        g.dispose();
        return combined;
    }

    private static ParsedRunArguments parseRunArguments(String[] args, int optionStartIndex) {
        List<String> positionalExtras = new ArrayList<>();
        Double fixedAlpha = null;
        Double ndviWetThreshold = null;
        Integer minNodeSizeOption = null;
        Integer performanceSampleLimit = null;
        Double trainValidationFraction = null;
        Path outputDir = null;

        for (int i = optionStartIndex; i < args.length; i++) {
            String arg = args[i];
            String lower = arg.toLowerCase(Locale.ROOT);
            if ("--alpha".equalsIgnoreCase(arg)) {
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("--alpha requires a numeric value");
                }
                fixedAlpha = Double.parseDouble(args[++i]);
            } else if (lower.startsWith("--alpha=")) {
                fixedAlpha = Double.parseDouble(arg.substring(arg.indexOf('=') + 1));
            } else if ("--ndvi-threshold".equalsIgnoreCase(arg)) {
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("--ndvi-threshold requires a numeric value");
                }
                ndviWetThreshold = Double.parseDouble(args[++i]);
            } else if (lower.startsWith("--ndvi-threshold=")) {
                ndviWetThreshold = Double.parseDouble(arg.substring(arg.indexOf('=') + 1));
            } else if ("--min-node-size".equalsIgnoreCase(arg)) {
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("--min-node-size requires an integer value");
                }
                minNodeSizeOption = Integer.parseInt(args[++i]);
            } else if (lower.startsWith("--min-node-size=")) {
                minNodeSizeOption = Integer.parseInt(arg.substring(arg.indexOf('=') + 1));
            } else if ("--train-validation-fraction".equalsIgnoreCase(arg)) {
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("--train-validation-fraction requires a numeric value in (0, 1)");
                }
                trainValidationFraction = Double.parseDouble(args[++i]);
            } else if (lower.startsWith("--train-validation-fraction=")) {
                trainValidationFraction = Double.parseDouble(arg.substring(arg.indexOf('=') + 1));
            } else if ("--perf-limit".equalsIgnoreCase(arg)) {
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("--perf-limit requires an integer value");
                }
                performanceSampleLimit = Integer.parseInt(args[++i]);
            } else if (lower.startsWith("--perf-limit=")) {
                performanceSampleLimit = Integer.parseInt(arg.substring(arg.indexOf('=') + 1));
            } else if ("--tree-print-limit".equalsIgnoreCase(arg)) {
                if (i + 1 >= args.length) throw new IllegalArgumentException("--tree-print-limit requires an integer");
                positionalExtras.add("__treePrintLimit=" + args[++i]);
            } else if (lower.startsWith("--tree-print-limit=")) {
                positionalExtras.add("__treePrintLimit=" + arg.substring(arg.indexOf('=') + 1));
            } else if ("--alpha-end".equalsIgnoreCase(arg)) {
                if (i + 1 >= args.length) throw new IllegalArgumentException("--alpha-end requires a numeric value");
                // will be resolved outside parseRunArguments via a second scan
                positionalExtras.add("__alphaEnd=" + args[++i]);
            } else if (lower.startsWith("--alpha-end=")) {
                positionalExtras.add("__alphaEnd=" + arg.substring(arg.indexOf('=') + 1));
            } else if ("--output-dir".equalsIgnoreCase(arg)) {
                if (i + 1 >= args.length) {
                    throw new IllegalArgumentException("--output-dir requires a path");
                }
                outputDir = Path.of(args[++i]);
            } else if (lower.startsWith("--output-dir=")) {
                outputDir = Path.of(arg.substring(arg.indexOf('=') + 1));
            } else {
                positionalExtras.add(arg);
            }
        }

        Integer treePrintLeafLimitParsed = null;
        for (String extra : positionalExtras) {
            if (extra.startsWith("__treePrintLimit=")) {
                treePrintLeafLimitParsed = Integer.parseInt(extra.substring("__treePrintLimit=".length()));
            }
        }
        return new ParsedRunArguments(positionalExtras, fixedAlpha, ndviWetThreshold, minNodeSizeOption, performanceSampleLimit, trainValidationFraction, outputDir, treePrintLeafLimitParsed);
    }

    private static List<Double> candidateAlphas(double start, double end, double step) {
        List<Double> values = new ArrayList<>();
        for (double value = start; value <= end + 1e-9; value += step) {
            values.add(Math.round(value * 10000.0) / 10000.0);
        }
        return values;
    }

    private static boolean isValid(RasterBundle rasters, int row, int col) {
        if (rasters.qa().get(row, col) != CLEAR_QA_PIXEL) {
            return false;
        }
        int b2 = rasters.b2().get(row, col);
        int b3 = rasters.b3().get(row, col);
        int red = rasters.b4().get(row, col);
        int nir = rasters.b5().get(row, col);
        int b6 = rasters.b6().get(row, col);
        int b7 = rasters.b7().get(row, col);
        return b2 != rasters.b2().noData()
                && b3 != rasters.b3().noData()
                && red != rasters.b4().noData()
                && nir != rasters.b5().noData()
                && b6 != rasters.b6().noData()
                && b7 != rasters.b7().noData()
                && red + nir != 0;
    }

    private static double ndvi(int red, int nir) {
        return (double) (nir - red) / (double) (nir + red);
    }

    private static int ndviClass(double ndvi, double ndviWetThreshold) {
        return ndvi >= ndviWetThreshold ? WET : DRY;
    }

    private static void printLabelLegend(double ndviWetThreshold) {
        System.out.println("  labels:");
        System.out.printf(Locale.US, "    0 low-NDVI: NDVI < %.4f%n", ndviWetThreshold);
        System.out.printf(Locale.US, "    1 high-NDVI: NDVI >= %.4f%n", ndviWetThreshold);
    }


    private static int detectRequiredPathCount(String[] args) {
        if (args.length >= 7 && !args[2].startsWith("--") && !looksNumeric(args[2])) {
            return 7;
        }
        return 2;
    }

    private static boolean looksNumeric(String s) {
        try {
            Double.parseDouble(s);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private static RasterBundle readRasterBundle(String[] args, int requiredPathCount, int noData) throws IOException {
        if (requiredPathCount == 7) {
            RasterBundle bundle = new RasterBundle(
                    TiffReader.readSingleBand(Path.of(args[0]), noData),
                    TiffReader.readSingleBand(Path.of(args[1]), noData),
                    TiffReader.readSingleBand(Path.of(args[2]), noData),
                    TiffReader.readSingleBand(Path.of(args[3]), noData),
                    TiffReader.readSingleBand(Path.of(args[4]), noData),
                    TiffReader.readSingleBand(Path.of(args[5]), noData),
                    TiffReader.readSingleBand(Path.of(args[6]), noData)
            );
            ensureSameShape(bundle.b2(), bundle.b3(), bundle.b4(), bundle.b5(), bundle.b6(), bundle.b7(), bundle.qa());
            return bundle;
        }
        SingleBandRaster b4 = TiffReader.readSingleBand(Path.of(args[0]), noData);
        SingleBandRaster b5 = TiffReader.readSingleBand(Path.of(args[1]), noData);
        SingleBandRaster qaAllClear = makeAllClearQa(b4);
        ensureSameShape(b4, b5);
        return new RasterBundle(b4, b5, b4, b5, b4, b5, qaAllClear);
    }

    private static SingleBandRaster makeAllClearQa(SingleBandRaster template) {
        int[] values = new int[template.width() * template.height()];
        java.util.Arrays.fill(values, CLEAR_QA_PIXEL);
        return new SingleBandRaster("synthetic_all_clear_QA_PIXEL", template.width(), template.height(), template.noData(), values);
    }

    private static void printRasterStats(RasterBundle rasters) {
        printStats("B2/blue feature", rasters.b2().statistics());
        printStats("B3/green feature", rasters.b3().statistics());
        printStats("B4/red label-only", rasters.b4().statistics());
        printStats("B5/NIR label-only", rasters.b5().statistics());
        printStats("B6/SWIR1 feature", rasters.b6().statistics());
        printStats("B7/SWIR2 feature", rasters.b7().statistics());
        printStats("QA_PIXEL mask", rasters.qa().statistics());
    }

    private static void ensureSameShape(SingleBandRaster first, SingleBandRaster... others) {
        for (SingleBandRaster other : others) {
            if (first.width() != other.width() || first.height() != other.height()) {
                throw new IllegalArgumentException("All rasters must have the same shape; "
                        + first.name() + " is " + first.width() + "x" + first.height()
                        + ", but " + other.name() + " is " + other.width() + "x" + other.height());
            }
        }
    }

    private static void writeRgbVisualizations(Path outputDir, RasterBundle rasters) throws IOException {
        ImageIO.write(makeRgbImage(rasters, rasters.b4(), rasters.b3(), rasters.b2()), "png",
                outputDir.resolve("true_color_rgb_b4_b3_b2.png").toFile());
        ImageIO.write(makeRgbImage(rasters, rasters.b5(), rasters.b4(), rasters.b3()), "png",
                outputDir.resolve("false_color_b5_b4_b3.png").toFile());
    }

    private static BufferedImage makeRgbImage(RasterBundle rasters, SingleBandRaster rBand, SingleBandRaster gBand, SingleBandRaster bBand) {
        int sourceWidth = rasters.width();
        int sourceHeight = rasters.height();
        int step = Math.max(1, (int) Math.ceil(Math.max(sourceWidth, sourceHeight) / 2000.0));
        int width = ceilDiv(sourceWidth, step);
        int height = ceilDiv(sourceHeight, step);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Stretch r = stretchFor(rBand, rasters.qa());
        Stretch g = stretchFor(gBand, rasters.qa());
        Stretch b = stretchFor(bBand, rasters.qa());
        int gray = new Color(55, 55, 55).getRGB();
        for (int y = 0, row = 0; row < sourceHeight; y++, row += step) {
            for (int x = 0, col = 0; col < sourceWidth; x++, col += step) {
                if (rasters.qa().get(row, col) != CLEAR_QA_PIXEL) {
                    image.setRGB(x, y, gray);
                    continue;
                }
                int rr = stretch(rBand.get(row, col), r);
                int gg = stretch(gBand.get(row, col), g);
                int bb = stretch(bBand.get(row, col), b);
                image.setRGB(x, y, new Color(rr, gg, bb).getRGB());
            }
        }
        return image;
    }

    private static Stretch stretchFor(SingleBandRaster band, SingleBandRaster qa) {
        int[] hist = new int[65536];
        long count = 0;
        for (int row = 0; row < band.height(); row++) {
            for (int col = 0; col < band.width(); col++) {
                int v = band.get(row, col);
                if (qa.get(row, col) != CLEAR_QA_PIXEL || v == band.noData()) continue;
                if (v < 0) v = 0;
                if (v >= hist.length) v = hist.length - 1;
                hist[v]++;
                count++;
            }
        }
        if (count == 0) return new Stretch(0, 1);
        int lowTarget = (int) Math.max(0, Math.floor(count * 0.02));
        int highTarget = (int) Math.min(count - 1, Math.floor(count * 0.98));
        int low = percentileFromHist(hist, lowTarget);
        int high = percentileFromHist(hist, highTarget);
        if (high <= low) high = low + 1;
        return new Stretch(low, high);
    }

    private static int percentileFromHist(int[] hist, int target) {
        long cumulative = 0;
        for (int i = 0; i < hist.length; i++) {
            cumulative += hist[i];
            if (cumulative > target) return i;
        }
        return hist.length - 1;
    }

    private static int stretch(int value, Stretch s) {
        double x = (value - s.low()) / (double) (s.high() - s.low());
        x = Math.max(0.0, Math.min(1.0, x));
        return (int) Math.round(x * 255.0);
    }

    public static void printUsage() {
        System.out.println("Usage:");
        System.out.println("Recommended seven-raster mode:");
        System.out.println("  ./gradlew run --args=\"LC09_L2SP_167041_20230317_20230320_02_T1_SR_B2.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B3.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B4.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B5.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B6.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B7.TIF LC09_L2SP_167041_20230317_20230320_02_T1_QA_PIXEL.TIF "
                + "[psuSize=98] [samplesPerPsu=1000] [trainPsuPerClass=8] "
                + "[validationPsuPerClass=2] [trainingAreaFraction=0.70 left-columns] [testEvalStep=1] "
                + "[noData=0] [alphaStart=0.02] [alphaEnd=1.00] [alphaStep=0.02] "
                + "[minNodeSize=100] [neighborDistance=6] [randomSeed=42] [ndviWetThreshold=0.10] "
                + "[--alpha value] [--ndvi-threshold value] [--min-node-size n] [--perf-limit n] [--output-dir path]\"");
        System.out.println();
        System.out.println("Fixed alpha example:");
        System.out.println("  ./gradlew run --args=\"LC09_L2SP_167041_20230317_20230320_02_T1_SR_B2.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B3.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B4.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B5.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B6.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B7.TIF LC09_L2SP_167041_20230317_20230320_02_T1_QA_PIXEL.TIF --alpha 0.26\"");
        System.out.println();
        System.out.println("If --alpha is omitted, the best alpha is searched on the validation set.");
        System.out.println("By default, the spatial split is left-right: left 70% columns are used for train/validation, right 30% columns are used only for test.");
        System.out.println("Use --train-validation-fraction to change the left-column fraction, for example --train-validation-fraction 0.70.");
        System.out.println("In recommended mode labels use B4/B5 NDVI, features are B2/B3/B6/B7, and QA_PIXEL must equal 21824.");
        System.out.println("Legacy two-raster mode is still accepted for old experiments, but it has label leakage and is not recommended.");
        System.out.println("Labels are binary NDVI pseudo-labels: low-NDVI (NDVI < threshold) and high-NDVI (NDVI >= threshold).");
    }

    private static void validateParameters(int psuSize,
                                           int samplesPerPsu,
                                           int trainPsuPerClass,
                                           int validationPsuPerClass,
                                           double trainingAreaFraction,
                                           int testEvalStep,
                                           double alphaStart,
                                           double alphaEnd,
                                           double alphaStep,
                                           int minNodeSize,
                                           double neighborDistance,
                                           double ndviWetThreshold,
                                           int performanceSampleLimit) {
        if (psuSize <= 0 || samplesPerPsu <= 0 || trainPsuPerClass <= 0 || validationPsuPerClass <= 0) {
            throw new IllegalArgumentException("PSU/sample parameters must be positive");
        }
        if (trainingAreaFraction <= 0.0 || trainingAreaFraction >= 1.0) {
            throw new IllegalArgumentException("trainingAreaFraction must be in (0, 1)");
        }
        if (testEvalStep <= 0) {
            throw new IllegalArgumentException("testEvalStep must be positive");
        }
        if (alphaStart < 0.0 || alphaEnd > 1.0 || alphaStart > alphaEnd || alphaStep <= 0.0) {
            throw new IllegalArgumentException("bad alpha sweep parameters");
        }
        if (minNodeSize <= 0) {
            throw new IllegalArgumentException("minNodeSize must be positive");
        }
        if (neighborDistance <= 0.0) {
            throw new IllegalArgumentException("neighborDistance must be positive");
        }
        if (ndviWetThreshold < -1.0 || ndviWetThreshold > 1.0) {
            throw new IllegalArgumentException("ndviWetThreshold must be in [-1, 1]");
        }
        if (performanceSampleLimit <= 0) {
            throw new IllegalArgumentException("performanceSampleLimit must be positive");
        }
    }

    private static void validateSamples(List<Sample> trainSamples, List<Sample> validationSamples, int minNodeSize) {
        LabelCounts trainCounts = countLabels(trainSamples);
        LabelCounts validationCounts = countLabels(validationSamples);
        if (trainSamples.size() < minNodeSize) {
            throw new IllegalStateException("Too few train samples: " + trainSamples.size());
        }
        if (validationSamples.isEmpty()) {
            throw new IllegalStateException("Validation samples are empty");
        }
        if (trainCounts.distinctClassCount() < 2) {
            throw new IllegalStateException("Train samples contain fewer than two NDVI classes: " + trainCounts);
        }
        if (validationCounts.distinctClassCount() < 2) {
            System.out.println("WARNING: validation samples contain fewer than two NDVI classes: " + validationCounts);
        }
    }

    private static void ensureSameShape(SingleBandRaster a, SingleBandRaster b) {
        if (a.width() != b.width() || a.height() != b.height()) {
            throw new IllegalArgumentException("Rasters must have the same width/height: "
                    + a.width() + "x" + a.height() + " vs " + b.width() + "x" + b.height());
        }
    }

    private static LabelCounts countLabels(List<Sample> samples) {
        long[] counts = new long[CLASS_COUNT];
        long other = 0;
        for (Sample sample : samples) {
            int label = sample.getLabel();
            if (label >= 0 && label < counts.length) {
                counts[label]++;
            } else {
                other++;
            }
        }
        return new LabelCounts(counts, other);
    }

    private static void printStats(String label, RasterStatistics statistics) {
        System.out.printf(Locale.US,
                "%s: %dx%d, valid=%d, noData=%d%n",
                label,
                statistics.width(), statistics.height(), statistics.validCount(), statistics.noDataCount());
    }

    private static void printBookConfusionMatrix(long[][] matrix) {
        System.out.println("                 Classified low-NDVI   Classified high-NDVI");
        System.out.printf(Locale.US, "  Truth low-NDVI          %18d%17d%n", matrix[DRY][DRY], matrix[DRY][WET]);
        System.out.printf(Locale.US, "  Truth high-NDVI         %18d%17d%n", matrix[WET][DRY], matrix[WET][WET]);
    }

    private static void printBbJoinCountTable(BbJoinCount dt, BbJoinCount sdt) {
        System.out.println("                    C4.5-DT              SDT");
        System.out.printf(Locale.US, "  Observed JC     %14d%17d%n", dt.observed(), sdt.observed());
        System.out.printf(Locale.US, "  Expected mu     %14.3f%17.3f%n", dt.expected(), sdt.expected());
        System.out.printf(Locale.US, "  Variance sigma2 %14.3e%17.3e%n", dt.variance(), sdt.variance());
        System.out.printf(Locale.US, "  (JC-mu)/sigma   %14.4f%17.4f%n", dt.normalized(), sdt.normalized());
    }

    private static void printPerformanceTable(List<PerformanceRow> rows) {
        System.out.println("  operation                         samples/px       edges        ms    notes");
        for (PerformanceRow row : rows) {
            System.out.printf(Locale.US, "  %-32s%10d%12d%10.3f    %s%n",
                    row.operation(), row.sampleCount(), row.edgeCount(), row.millis(), row.notes());
        }
    }

    private static void writeConfusionCsv(Path path, long[][] matrix) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("truth,predicted_low_ndvi,predicted_high_ndvi\n");
        sb.append("low_ndvi,").append(matrix[DRY][DRY]).append(',').append(matrix[DRY][WET]).append('\n');
        sb.append("high_ndvi,").append(matrix[WET][DRY]).append(',').append(matrix[WET][WET]).append('\n');
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeBbCsv(Path path, BbJoinCount dt, BbJoinCount sdt) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("metric,C4.5-DT,SDT\n");
        sb.append("Observed JC,").append(dt.observed()).append(',').append(sdt.observed()).append('\n');
        sb.append("Expected mu,").append(dt.expected()).append(',').append(sdt.expected()).append('\n');
        sb.append("Variance sigma2,").append(dt.variance()).append(',').append(sdt.variance()).append('\n');
        sb.append("(JC-mu)/sigma,").append(dt.normalized()).append(',').append(sdt.normalized()).append('\n');
        sb.append("valid nodes,").append(dt.validNodes()).append(',').append(sdt.validNodes()).append('\n');
        sb.append("high-NDVI nodes,").append(dt.wetNodes()).append(',').append(sdt.wetNodes()).append('\n');
        sb.append("rook edges,").append(dt.edges()).append(',').append(sdt.edges()).append('\n');
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writePerformanceCsv(Path path, List<PerformanceRow> rows) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("operation,sample_or_pixel_count,edge_count,milliseconds,notes\n");
        for (PerformanceRow row : rows) {
            sb.append(csv(row.operation())).append(',')
                    .append(row.sampleCount()).append(',')
                    .append(row.edgeCount()).append(',')
                    .append(row.millis()).append(',')
                    .append(csv(row.notes())).append('\n');
        }
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeAlphaTuningCsv(Path path, List<AlphaTuner.EvaluationPoint> evaluations) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("alpha,train_error,validation_error\n");
        for (AlphaTuner.EvaluationPoint point : evaluations) {
            sb.append(point.alpha()).append(',')
                    .append(point.trainingError()).append(',')
                    .append(point.validationError()).append('\n');
        }
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static String csv(String value) {
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    private static String formatClassArray(int[] values) {
        return "[low-NDVI=" + values[DRY] + ", high-NDVI=" + values[WET] + "]";
    }

    private static String percent(double value) {
        return String.format(Locale.US, "%.1f%%", value * 100.0);
    }

    private static String labelName(int label) {
        return label == WET ? "high-NDVI" : "low-NDVI";
    }

    private static String yesNo(boolean value) {
        return value ? "YES" : "NO";
    }

    private static String fmt4(double value) {
        return String.format(Locale.US, "%.4f", value);
    }

    private static String fmt6(double value) {
        return String.format(Locale.US, "%.6f", value);
    }

    private static int ceilDiv(int value, int divisor) {
        return (value + divisor - 1) / divisor;
    }

    private static int[] filledIntArray(int length, int value) {
        int[] array = new int[length];
        if (value != 0) {
            for (int i = 0; i < array.length; i++) {
                array[i] = value;
            }
        }
        return array;
    }

    private static <T> TimedResult<T> time(String label, ThrowingSupplier<T> supplier) {
        long started = System.nanoTime();
        try {
            T value = supplier.get();
            long elapsed = System.nanoTime() - started;
            return new TimedResult<>(value, elapsed / 1_000_000.0);
        } catch (Exception e) {
            throw new RuntimeException("Failed during " + label, e);
        }
    }

    @FunctionalInterface
    private interface ThrowingSupplier<T> {
        T get() throws Exception;
    }

    private record Area(int rowStart, int rowEnd, int colStart, int colEnd) {
    }

    private record RasterBundle(SingleBandRaster b2, SingleBandRaster b3, SingleBandRaster b4, SingleBandRaster b5, SingleBandRaster b6, SingleBandRaster b7, SingleBandRaster qa) {
        int width() { return b2.width(); }
        int height() { return b2.height(); }
    }

    private record Stretch(int low, int high) {}

    private record ParsedRunArguments(List<String> positionalExtras,
                                      Double fixedAlpha,
                                      Double ndviWetThreshold,
                                      Integer minNodeSizeOption,
                                      Integer performanceSampleLimit,
                                      Double trainValidationFraction,
                                      Path outputDir,
                                      Integer treePrintLeafLimit) {
    }

    private record PixelRef(int row, int col) {
    }

    private record PsuInfo(int rowStart,
                           int rowEnd,
                           int colStart,
                           int colEnd,
                           int[] classCounts,
                           int majorityClass) {
    }

    private record BookStyleSplit(List<Sample> trainSamples,
                                  List<Sample> validationSamples,
                                  int[] trainPsuCounts,
                                  int[] validationPsuCounts) {
    }

    private record StreamingEvaluation(int mapWidth,
                                       int mapHeight,
                                       long validPixels,
                                       long skippedPixels,
                                       long errors,
                                       long[] predictedCounts,
                                       long[] truthCounts,
                                       long[][] confusionMatrix,
                                       int[] truthLabels,
                                       int[] predictedLabels) {
        double errorRate() {
            return validPixels == 0 ? 0.0 : (double) errors / (double) validPixels;
        }
    }

    private record BbJoinCount(long observed,
                               double expected,
                               double variance,
                               double normalized,
                               long validNodes,
                               long wetNodes,
                               long edges) {
    }

    private record PerformanceRow(String operation,
                                  int sampleCount,
                                  int edgeCount,
                                  double millis,
                                  String notes) {
    }

    private static final class TreeStatsAccumulator {
        int nodes = 0;
        int leaves = 0;
        int maxDepth = 0;
        int minLeafSize = Integer.MAX_VALUE;
        int maxLeafSize = 0;
    }

    private record TreeStats(int nodes, int leaves, int maxDepth, int minLeafSize, int maxLeafSize) {
    }

    private record TimedResult<T>(T value, double millis) {
    }

    private record LabelCounts(long[] labels, long other) {
        int distinctClassCount() {
            int count = 0;
            for (long labelCount : labels) {
                if (labelCount > 0) {
                    count++;
                }
            }
            if (other > 0) {
                count++;
            }
            return count;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("{");
            sb.append("low-NDVI=").append(labels[DRY]).append(", high-NDVI=").append(labels[WET]);
            if (other > 0) {
                sb.append(", other=").append(other);
            }
            return sb.append("}").toString();
        }
    }
}
