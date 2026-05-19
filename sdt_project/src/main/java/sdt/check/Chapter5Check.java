package sdt.check;

import sdt.ch5.FocalNeighborhoodMode;
import sdt.ch5.FocalSpatialDecisionTree;
import sdt.io.SingleBandRaster;
import sdt.io.TiffReader;
import sdt.model.Sample;
import sdt.tree.C45DecisionTree;
import sdt.tree.TreeClassifier;
import sdt.viz.ChartUtils;

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

public final class Chapter5Check {
    private static final int DRY = 0;
    private static final int WET = 1;
    private static final int CLASS_COUNT = 2;
    private static final int CLEAR_QA_PIXEL = 21824;

    private Chapter5Check() {}

    public static void run(String[] args) throws IOException {
        if (args.length < 2) {
            printUsage();
            return;
        }
        Parsed p = parse(args);
        RasterBundle rasters = readRasterBundle(args, p.requiredPathCount, p.noData);
        Files.createDirectories(p.outputDir);

        int trainingEndRow = Math.max(1, Math.min(rasters.height() - 1, (int) Math.round(rasters.height() * p.trainingFraction)));
        Area trainArea = new Area(0, trainingEndRow, 0, rasters.width());
        Area testArea = new Area(trainingEndRow, rasters.height(), 0, rasters.width());
        BookSplit split = buildBookStyleSplit(rasters, trainArea, p.psuSize, p.trainPsuPerClass, p.validationPsuPerClass, p.samplesPerPsu, p.seed, p.ndviThreshold, p.includeNdviFeature);
        List<Sample> evalSamples = sampleEvaluationArea(rasters, testArea, p.evalStep, p.maxEvalSamples, p.seed + 9, p.ndviThreshold, p.includeNdviFeature);

        System.out.println("\nLTDT vs FTSDT: Focal-Test-Based Spatial Decision Tree");
        System.out.println("  truth note: pseudo-labels only, high-NDVI = NDVI >= " + p.ndviThreshold);
        System.out.println("  split: top rows [0, " + trainingEndRow + ") for train/validation, bottom rows [" + trainingEndRow + ", " + rasters.height() + ") for evaluation");
        System.out.println("  train=" + split.train.size() + " validation=" + split.validation.size() + " eval=" + evalSamples.size());
        System.out.println("  evalStep=" + p.evalStep + " maxEvalSamples=" + p.maxEvalSamples);
        System.out.println("  features=" + (p.includeNdviFeature ? "B2, B3, B6, B7, NDVI" : "B2, B3, B6, B7 (B4/B5 excluded to avoid pseudo-label leakage)"));
        System.out.println("  outputDir=" + p.outputDir.toAbsolutePath());

        List<PerformanceRow> perf = new ArrayList<>();
        runChapter5(split.train, split.validation, evalSamples, p, perf);
        writePerformance(p.outputDir.resolve("performance.csv"), perf);
        writePerformanceChart(p.outputDir.resolve("performance_chart.png"), perf);
        System.out.println("\nPerformance rows written: " + p.outputDir.resolve("performance.csv").toAbsolutePath());
        System.out.println("Performance chart written: " + p.outputDir.resolve("performance_chart.png").toAbsolutePath());
    }

    private static void runChapter5(List<Sample> train, List<Sample> validation, List<Sample> eval, Parsed p, List<PerformanceRow> perf) throws IOException {
        System.out.println("\nFocal-Test-Based Spatial Decision Tree:");
        long t0 = System.nanoTime();
        C45DecisionTree ltdt = new C45DecisionTree(p.minNodeSize);
        ltdt.fit(train);
        perf.add(new PerformanceRow("LTDT (C4.5) train", train.size(), millisSince(t0), "local-test C4.5 baseline"));

        t0 = System.nanoTime();
        FocalSpatialDecisionTree fixed = new FocalSpatialDecisionTree(p.sMax, p.minNodeSize, FocalNeighborhoodMode.FIXED);
        fixed.fit(train);
        perf.add(new PerformanceRow("FTSDT-fixed train", train.size(), millisSince(t0), "Algorithm 6 refined focal test, fixed windows"));

        t0 = System.nanoTime();
        FocalSpatialDecisionTree adaptive = new FocalSpatialDecisionTree(p.sMax, p.minNodeSize, FocalNeighborhoodMode.ADAPTIVE);
        adaptive.fit(train);
        perf.add(new PerformanceRow("FTSDT-adaptive train", train.size(), millisSince(t0), "Algorithm 6 refined focal test, adaptive windows"));

        ClassificationMetrics ltdtEval = evaluateTree(ltdt, eval, p.evalStep);
        ClassificationMetrics fixedEval = evaluateBatch(fixed.predictBatch(eval), eval, p.evalStep);
        ClassificationMetrics adaptiveEval = evaluateBatch(adaptive.predictBatch(eval), eval, p.evalStep);

        System.out.println("  LTDT          " + ltdtEval.summary());
        System.out.println("  FTSDT-fixed   " + fixedEval.summary() + " tree=" + FocalSpatialDecisionTree.statsString(fixed.getRoot()));
        System.out.println("  FTSDT-adapt   " + adaptiveEval.summary() + " tree=" + FocalSpatialDecisionTree.statsString(adaptive.getRoot()));

        List<NamedMetrics> rows = List.of(
                new NamedMetrics("LTDT", ltdtEval),
                new NamedMetrics("FTSDT-fixed", fixedEval),
                new NamedMetrics("FTSDT-adaptive", adaptiveEval)
        );
        writeMetricsCsv(p.outputDir.resolve("classification.csv"), rows);
        writeMetricsChart(p.outputDir.resolve("classification_chart.png"), "Classification Metrics", rows);
        System.out.println("  wrote " + p.outputDir.resolve("classification.csv").toAbsolutePath());
        System.out.println("  chart " + p.outputDir.resolve("classification_chart.png").toAbsolutePath());
    }

    private static ClassificationMetrics evaluateTree(TreeClassifier tree, List<Sample> eval, int latticeStep) {
        int[] pred = new int[eval.size()];
        for (int i = 0; i < eval.size(); i++) pred[i] = tree.predict(eval.get(i).getFeatures());
        return evaluateBatch(pred, eval, latticeStep);
    }

    private static ClassificationMetrics evaluateBatch(int[] pred, List<Sample> eval, int latticeStep) {
        long[][] cm = new long[CLASS_COUNT][CLASS_COUNT];
        int[] labels = new int[eval.size()];
        int errors = 0;
        for (int i = 0; i < eval.size(); i++) {
            int truth = eval.get(i).getLabel();
            int p = pred[i];
            labels[i] = p;
            if (p >= 0 && p < CLASS_COUNT) cm[truth][p]++;
            if (p != truth) errors++;
        }
        double precision = cm[DRY][WET] + cm[WET][WET] == 0 ? 0.0 : (double) cm[WET][WET] / (cm[DRY][WET] + cm[WET][WET]);
        double recall = cm[WET][DRY] + cm[WET][WET] == 0 ? 0.0 : (double) cm[WET][WET] / (cm[WET][DRY] + cm[WET][WET]);
        double f = precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
        return new ClassificationMetrics(cm, errors / (double) eval.size(), precision, recall, f, gammaIndex(eval, labels, latticeStep));
    }

    private static double gammaIndex(List<Sample> samples, int[] labels, int latticeStep) {
        int n = samples.size();
        if (n == 0) return Double.NaN;

        double mean = 0.0;
        for (int l : labels) mean += l;
        mean /= n;

        double sumVar = 0.0;
        for (int l : labels) { double z = l - mean; sumVar += z * z; }
        if (sumVar == 0.0) return Double.NaN;

        Map<Long, Integer> byCell = new HashMap<>(n * 2);
        for (int i = 0; i < n; i++)
            byCell.put(cellKey(samples.get(i).getRow(), samples.get(i).getCol()), i);

        int step = Math.max(1, latticeStep);
        double sumW = 0.0, sumCross = 0.0;

        for (int i = 0; i < n; i++) {
            Sample s = samples.get(i);
            double zi = labels[i] - mean;
            for (int dr = -step; dr <= step; dr += step) {
                for (int dc = -step; dc <= step; dc += step) {
                    if (dr == 0 && dc == 0) continue;
                    Integer j = byCell.get(cellKey(s.getRow() + dr, s.getCol() + dc));
                    if (j == null) continue;
                    sumW++;
                    sumCross += zi * (labels[j] - mean);
                }
            }
        }

        if (sumW == 0.0) return Double.NaN;
        return (n * sumCross) / (sumW * sumVar);
    }

    private static BookSplit buildBookStyleSplit(RasterBundle rasters, Area area, int psuSize,
                                                 int trainPsuPerClass, int validationPsuPerClass, int samplesPerPsu,
                                                 long seed, double threshold, boolean includeNdvi) {
        List<Psu> psus = scanPsus(rasters, area, psuSize, threshold);
        Map<Integer, List<Psu>> byClass = new HashMap<>(); byClass.put(DRY, new ArrayList<>()); byClass.put(WET, new ArrayList<>());
        for (Psu psu : psus) byClass.get(psu.majority).add(psu);
        Random random = new Random(seed);
        List<Sample> train = new ArrayList<>(), val = new ArrayList<>();
        int id = 0;
        for (int label = 0; label < CLASS_COUNT; label++) {
            List<Psu> list = byClass.get(label);
            list.sort((a, b) -> {
                int byCol = Integer.compare(a.cs, b.cs);
                if (byCol != 0) return byCol;
                return Integer.compare(a.rs, b.rs);
            });
            int t = Math.min(trainPsuPerClass, list.size());
            int v = Math.min(validationPsuPerClass, Math.max(0, list.size() - t));
            for (int i = 0; i < t; i++) { List<Sample> smp = samplePsu(rasters, list.get(i), samplesPerPsu, id, random, threshold, includeNdvi); id += smp.size(); train.addAll(smp); }
            for (int i = t; i < t + v; i++) { List<Sample> smp = samplePsu(rasters, list.get(i), samplesPerPsu, id, random, threshold, includeNdvi); id += smp.size(); val.addAll(smp); }
        }
        return new BookSplit(train, val);
    }

    private static List<Psu> scanPsus(RasterBundle rasters, Area area, int psuSize, double threshold) {
        List<Psu> out = new ArrayList<>();
        for (int rs = area.rs; rs < area.re; rs += psuSize) {
            for (int cs = area.cs; cs < area.ce; cs += psuSize) {
                int re = Math.min(rs + psuSize, area.re), ce = Math.min(cs + psuSize, area.ce);
                int[] counts = new int[2];
                for (int r = rs; r < re; r++) for (int c = cs; c < ce; c++) if (valid(rasters, r, c)) counts[label(rasters, r, c, threshold)]++;
                if (counts[0] + counts[1] > 0) out.add(new Psu(rs, re, cs, ce, counts[WET] > counts[DRY] ? WET : DRY));
            }
        }
        return out;
    }

    private static List<Sample> samplePsu(RasterBundle rasters, Psu psu, int n, int firstId, Random rnd, double threshold, boolean includeNdvi) {
        List<int[]> cells = new ArrayList<>();
        for (int r = psu.rs; r < psu.re; r++) for (int c = psu.cs; c < psu.ce; c++) if (valid(rasters, r, c)) cells.add(new int[]{r, c});
        Collections.shuffle(cells, rnd);
        List<Sample> out = new ArrayList<>(); int id = firstId;
        for (int i = 0; i < Math.min(n, cells.size()); i++) out.add(toSample(rasters, cells.get(i)[0], cells.get(i)[1], id++, threshold, includeNdvi));
        return out;
    }

    private static List<Sample> sampleEvaluationArea(RasterBundle rasters, Area area, int step, int max, long seed, double threshold, boolean includeNdvi) {
        List<Sample> out = new ArrayList<>(); int id = 10_000_000;
        for (int r = area.rs; r < area.re; r += step) for (int c = area.cs; c < area.ce; c += step) if (valid(rasters, r, c)) out.add(toSample(rasters, r, c, id++, threshold, includeNdvi));
        if (max > 0 && out.size() > max) { Collections.shuffle(out, new Random(seed)); return new ArrayList<>(out.subList(0, max)); }
        return out;
    }

    private static Sample toSample(RasterBundle rasters, int r, int c, int id, double threshold, boolean includeNdvi) {
        double red = rasters.b4().get(r, c), nir = rasters.b5().get(r, c), ndvi = ndvi(red, nir);
        double[] features = includeNdvi
                ? new double[]{rasters.b2().get(r, c), rasters.b3().get(r, c), rasters.b6().get(r, c), rasters.b7().get(r, c), ndvi}
                : new double[]{rasters.b2().get(r, c), rasters.b3().get(r, c), rasters.b6().get(r, c), rasters.b7().get(r, c)};
        return new Sample(id, r, c, features, ndvi >= threshold ? WET : DRY);
    }

    private static boolean valid(RasterBundle rasters, int r, int c) {
        int red = rasters.b4().get(r, c), nir = rasters.b5().get(r, c);
        return rasters.qa().get(r, c) == CLEAR_QA_PIXEL
                && rasters.b2().get(r, c) != rasters.b2().noData()
                && rasters.b3().get(r, c) != rasters.b3().noData()
                && red != rasters.b4().noData()
                && nir != rasters.b5().noData()
                && rasters.b6().get(r, c) != rasters.b6().noData()
                && rasters.b7().get(r, c) != rasters.b7().noData()
                && red + nir != 0;
    }
    private static int label(RasterBundle rasters, int r, int c, double t) { return ndvi(rasters.b4().get(r, c), rasters.b5().get(r, c)) >= t ? WET : DRY; }
    private static double ndvi(double red, double nir) { return (nir - red) / (nir + red); }
    private static long cellKey(int row, int col) { return (((long) row) << 32) ^ (col & 0xffffffffL); }
    private static double millisSince(long t0) { return (System.nanoTime() - t0) / 1_000_000.0; }

    private static Parsed parse(String[] args) {
        Parsed p = new Parsed();
        p.requiredPathCount = detectRequiredPathCount(args);
        if (args.length < p.requiredPathCount) throw new IllegalArgumentException("not enough raster paths");
        for (int i = p.requiredPathCount; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--out", "--output-dir" -> p.outputDir = Path.of(args[++i]);
                case "--ndvi-threshold" -> p.ndviThreshold = Double.parseDouble(args[++i]);
                case "--eval-step" -> p.evalStep = Integer.parseInt(args[++i]);
                case "--max-eval" -> p.maxEvalSamples = Integer.parseInt(args[++i]);
                case "--smax" -> p.sMax = Integer.parseInt(args[++i]);
                case "--min-node-size" -> p.minNodeSize = Integer.parseInt(args[++i]);
                case "--psu-size" -> p.psuSize = Integer.parseInt(args[++i]);
                case "--samples-per-psu" -> p.samplesPerPsu = Integer.parseInt(args[++i]);
                case "--train-psu-per-class" -> p.trainPsuPerClass = Integer.parseInt(args[++i]);
                case "--validation-psu-per-class" -> p.validationPsuPerClass = Integer.parseInt(args[++i]);
                case "--training-fraction" -> p.trainingFraction = Double.parseDouble(args[++i]);
                case "--include-ndvi-feature" -> p.includeNdviFeature = true;
                case "--no-eval-cap" -> p.maxEvalSamples = 0;
                default -> { }
            }
        }
        return p;
    }

    private static int detectRequiredPathCount(String[] args) {
        if (args.length >= 7 && !args[2].startsWith("--") && !looksNumeric(args[2])) return 7;
        return 2;
    }

    private static boolean looksNumeric(String s) {
        try { Double.parseDouble(s); return true; } catch (NumberFormatException e) { return false; }
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
                    TiffReader.readSingleBand(Path.of(args[6]), noData));
            ensureSameShape(bundle.b2(), bundle.b3(), bundle.b4(), bundle.b5(), bundle.b6(), bundle.b7(), bundle.qa());
            return bundle;
        }
        SingleBandRaster b4 = TiffReader.readSingleBand(Path.of(args[0]), noData);
        SingleBandRaster b5 = TiffReader.readSingleBand(Path.of(args[1]), noData);
        ensureSameShape(b4, b5);
        int[] qa = new int[b4.width() * b4.height()];
        java.util.Arrays.fill(qa, CLEAR_QA_PIXEL);
        SingleBandRaster syntheticQa = new SingleBandRaster("synthetic_all_clear_QA_PIXEL", b4.width(), b4.height(), noData, qa);
        return new RasterBundle(b4, b5, b4, b5, b4, b5, syntheticQa);
    }

    private static void ensureSameShape(SingleBandRaster first, SingleBandRaster... others) {
        for (SingleBandRaster other : others) {
            if (first.width() != other.width() || first.height() != other.height()) throw new IllegalArgumentException("rasters must have same shape");
        }
    }

    private static void writeMetricsCsv(Path path, List<NamedMetrics> rows) throws IOException {
        StringBuilder sb = new StringBuilder("model,truth_low_ndvi_pred_low_ndvi,truth_low_ndvi_pred_high_ndvi,truth_high_ndvi_pred_low_ndvi,truth_high_ndvi_pred_high_ndvi,error,precision_high_ndvi,recall_high_ndvi,f_score_high_ndvi,gamma_queen\n");
        for (NamedMetrics r : rows) {
            long[][] cm = r.metrics.cm;
            sb.append(r.name).append(',').append(cm[0][0]).append(',').append(cm[0][1]).append(',').append(cm[1][0]).append(',').append(cm[1][1]).append(',')
                    .append(r.metrics.error).append(',').append(r.metrics.precision).append(',').append(r.metrics.recall).append(',').append(r.metrics.fScore).append(',').append(r.metrics.gamma).append('\n');
        }
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writePerformance(Path path, List<PerformanceRow> rows) throws IOException {
        StringBuilder sb = new StringBuilder("operation,samples,ms,notes\n");
        for (PerformanceRow r : rows) sb.append(r.operation).append(',').append(r.samples).append(',').append(r.ms).append(',').append(r.notes.replace(',', ';')).append('\n');
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    private static void writeMetricsChart(Path path, String title, List<NamedMetrics> rows) throws IOException {
        String[] groups = new String[rows.size()];
        double[][] values = new double[rows.size()][4];
        for (int i = 0; i < rows.size(); i++) {
            NamedMetrics row = rows.get(i);
            groups[i] = row.name;
            values[i][0] = row.metrics.error;
            values[i][1] = row.metrics.precision;
            values[i][2] = row.metrics.recall;
            values[i][3] = row.metrics.fScore;
        }
        ChartUtils.writeGroupedBarChart(path, title, "score / error", groups,
                new String[]{"error", "precision", "recall", "F-score"}, values);
    }

    private static void writePerformanceChart(Path path, List<PerformanceRow> rows) throws IOException {
        String[] labels = new String[rows.size()];
        double[] values = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            labels[i] = rows.get(i).operation;
            values[i] = rows.get(i).ms;
        }
        ChartUtils.writeBarChart(path, "Performance", "milliseconds", labels, values);
    }

    private static void printUsage() {
        System.out.println("Usage: ./gradlew run --args=\"LC09_L2SP_167041_20230317_20230320_02_T1_SR_B2.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B3.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B4.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B5.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B6.TIF LC09_L2SP_167041_20230317_20230320_02_T1_SR_B7.TIF LC09_L2SP_167041_20230317_20230320_02_T1_QA_PIXEL.TIF --chapter5 [options]\"");
    }

    private static final class Parsed {
        int requiredPathCount = 2; int noData = 0; int psuSize = 98, samplesPerPsu = 1000, trainPsuPerClass = 8, validationPsuPerClass = 2;
        double trainingFraction = 0.70, ndviThreshold = 0.10; long seed = 42; int evalStep = 4, maxEvalSamples = 200_000; boolean includeNdviFeature = false;
        int sMax = 5, minNodeSize = 100; Path outputDir = Path.of("outputs_2");
    }
    private record RasterBundle(SingleBandRaster b2, SingleBandRaster b3, SingleBandRaster b4, SingleBandRaster b5, SingleBandRaster b6, SingleBandRaster b7, SingleBandRaster qa) {
        int width() { return b2.width(); }
        int height() { return b2.height(); }
    }
    private record Area(int rs, int re, int cs, int ce) {}
    private record Psu(int rs, int re, int cs, int ce, int majority) {}
    private record BookSplit(List<Sample> train, List<Sample> validation) {}
    private record PerformanceRow(String operation, int samples, double ms, String notes) {}
    private record NamedMetrics(String name, ClassificationMetrics metrics) {}
    private record ClassificationMetrics(long[][] cm, double error, double precision, double recall, double fScore, double gamma) {
        String summary() { return String.format(Locale.US, "error=%.4f precision=%.4f recall=%.4f F=%.4f gamma=%.4f CM=[[%,d,%,d],[%,d,%,d]]", error, precision, recall, fScore, gamma, cm[0][0], cm[0][1], cm[1][0], cm[1][1]); }
    }
}
