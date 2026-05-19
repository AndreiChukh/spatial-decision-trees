package sdt.viz;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;

public final class ChartUtils {
    private static final int W = 1100;
    private static final int H = 700;
    private static final int LEFT = 105;
    private static final int RIGHT = 45;
    private static final int TOP = 80;
    private static final int BOTTOM = 130;

    private static final Color[] PALETTE = new Color[]{
            new Color(54, 102, 187),
            new Color(220, 105, 39),
            new Color(64, 150, 82),
            new Color(170, 80, 170),
            new Color(50, 150, 180),
            new Color(180, 140, 55)
    };

    private ChartUtils() {}

    public static void writeBarChart(Path path,
                                     String title,
                                     String yLabel,
                                     String[] labels,
                                     double[] values) throws IOException {
        if (labels.length != values.length) throw new IllegalArgumentException("labels/values length mismatch");
        BufferedImage image = canvas(title, yLabel);
        Graphics2D g = image.createGraphics();
        setup(g);
        drawAxes(g);
        double max = max(values);
        if (max <= 0.0) max = 1.0;
        max *= 1.10;
        drawYTicks(g, 0.0, max);
        int plotW = W - LEFT - RIGHT;
        int plotH = H - TOP - BOTTOM;
        int n = Math.max(1, labels.length);
        int slot = Math.max(1, plotW / n);
        int barW = Math.max(14, Math.min(70, (int) (slot * 0.62)));
        for (int i = 0; i < labels.length; i++) {
            int x = LEFT + i * slot + (slot - barW) / 2;
            int barH = (int) Math.round((values[i] / max) * plotH);
            int y = TOP + plotH - barH;
            g.setColor(PALETTE[i % PALETTE.length]);
            g.fillRect(x, y, barW, barH);
            g.setColor(Color.DARK_GRAY);
            g.drawRect(x, y, barW, barH);
            drawRotatedLabel(g, labels[i], x + barW / 2, H - BOTTOM + 22);
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
            String v = fmt(values[i]);
            FontMetrics fm = g.getFontMetrics();
            g.drawString(v, x + (barW - fm.stringWidth(v)) / 2, Math.max(TOP + 14, y - 6));
        }
        g.dispose();
        ImageIO.write(image, "png", path.toFile());
    }

    public static void writeGroupedBarChart(Path path,
                                            String title,
                                            String yLabel,
                                            String[] groups,
                                            String[] series,
                                            double[][] values) throws IOException {
        if (groups.length != values.length) throw new IllegalArgumentException("groups/values length mismatch");
        for (double[] row : values) if (row.length != series.length) throw new IllegalArgumentException("series/values length mismatch");
        BufferedImage image = canvas(title, yLabel);
        Graphics2D g = image.createGraphics();
        setup(g);
        drawAxes(g);
        double max = 0.0;
        for (double[] row : values) for (double v : row) max = Math.max(max, v);
        if (max <= 0.0) max = 1.0;
        max *= 1.15;
        drawYTicks(g, 0.0, max);

        int plotW = W - LEFT - RIGHT;
        int plotH = H - TOP - BOTTOM;
        int groupSlot = Math.max(1, plotW / Math.max(1, groups.length));
        int groupInner = (int) (groupSlot * 0.72);
        int barW = Math.max(8, groupInner / Math.max(1, series.length));
        for (int i = 0; i < groups.length; i++) {
            int groupX = LEFT + i * groupSlot + (groupSlot - groupInner) / 2;
            for (int j = 0; j < series.length; j++) {
                int x = groupX + j * barW;
                int barH = (int) Math.round((values[i][j] / max) * plotH);
                int y = TOP + plotH - barH;
                g.setColor(PALETTE[j % PALETTE.length]);
                g.fillRect(x, y, Math.max(1, barW - 3), barH);
                g.setColor(Color.DARK_GRAY);
                g.drawRect(x, y, Math.max(1, barW - 3), barH);
            }
            drawRotatedLabel(g, groups[i], groupX + groupInner / 2, H - BOTTOM + 22);
        }
        drawLegend(g, series, W - RIGHT - 230, TOP - 35);
        g.dispose();
        ImageIO.write(image, "png", path.toFile());
    }

    public static void writeLineChart(Path path,
                                      String title,
                                      String yLabel,
                                      String[] seriesNames,
                                      double[] x,
                                      double[][] ys) throws IOException {
        if (ys.length != seriesNames.length) throw new IllegalArgumentException("series length mismatch");
        for (double[] y : ys) if (y.length != x.length) throw new IllegalArgumentException("x/y length mismatch");
        BufferedImage image = canvas(title, yLabel);
        Graphics2D g = image.createGraphics();
        setup(g);
        drawAxes(g);
        double xMin = min(x), xMax = max(x);
        if (xMax <= xMin) xMax = xMin + 1.0;
        double yMin = Double.POSITIVE_INFINITY, yMax = Double.NEGATIVE_INFINITY;
        for (double[] y : ys) {
            yMin = Math.min(yMin, min(y));
            yMax = Math.max(yMax, max(y));
        }
        if (!Double.isFinite(yMin) || !Double.isFinite(yMax) || yMax <= yMin) { yMin = 0.0; yMax = 1.0; }
        double pad = (yMax - yMin) * 0.10;
        yMin = Math.max(0.0, yMin - pad);
        yMax = yMax + pad;
        drawYTicks(g, yMin, yMax);
        drawXTicks(g, xMin, xMax);

        int plotW = W - LEFT - RIGHT;
        int plotH = H - TOP - BOTTOM;
        g.setStroke(new BasicStroke(2.5f));
        for (int s = 0; s < ys.length; s++) {
            g.setColor(PALETTE[s % PALETTE.length]);
            for (int i = 1; i < x.length; i++) {
                int x1 = LEFT + (int) Math.round(((x[i - 1] - xMin) / (xMax - xMin)) * plotW);
                int y1 = TOP + plotH - (int) Math.round(((ys[s][i - 1] - yMin) / (yMax - yMin)) * plotH);
                int x2 = LEFT + (int) Math.round(((x[i] - xMin) / (xMax - xMin)) * plotW);
                int y2 = TOP + plotH - (int) Math.round(((ys[s][i] - yMin) / (yMax - yMin)) * plotH);
                g.drawLine(x1, y1, x2, y2);
            }
            for (int i = 0; i < x.length; i++) {
                int px = LEFT + (int) Math.round(((x[i] - xMin) / (xMax - xMin)) * plotW);
                int py = TOP + plotH - (int) Math.round(((ys[s][i] - yMin) / (yMax - yMin)) * plotH);
                g.fillOval(px - 3, py - 3, 7, 7);
            }
        }
        g.setStroke(new BasicStroke(1.0f));
        drawLegend(g, seriesNames, W - RIGHT - 260, TOP - 35);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 13));
        g.setColor(Color.DARK_GRAY);
        g.drawString("x", LEFT + plotW / 2, H - 30);
        g.dispose();
        ImageIO.write(image, "png", path.toFile());
    }

    private static BufferedImage canvas(String title, String yLabel) {
        BufferedImage image = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        setup(g);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, W, H);
        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 24));
        g.drawString(title, LEFT, 45);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 14));
        g.drawString(yLabel, 20, TOP - 20);
        g.dispose();
        return image;
    }

    private static void setup(Graphics2D g) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
    }

    private static void drawAxes(Graphics2D g) {
        int plotH = H - TOP - BOTTOM;
        int plotW = W - LEFT - RIGHT;
        g.setColor(Color.LIGHT_GRAY);
        g.fillRect(LEFT, TOP, plotW, plotH);
        g.setColor(Color.WHITE);
        for (int i = 0; i <= 5; i++) {
            int y = TOP + (plotH * i) / 5;
            g.drawLine(LEFT, y, LEFT + plotW, y);
        }
        g.setColor(Color.BLACK);
        g.drawRect(LEFT, TOP, plotW, plotH);
    }

    private static void drawYTicks(Graphics2D g, double min, double max) {
        int plotH = H - TOP - BOTTOM;
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        g.setColor(Color.DARK_GRAY);
        for (int i = 0; i <= 5; i++) {
            double v = max - (max - min) * i / 5.0;
            int y = TOP + (plotH * i) / 5;
            String label = fmt(v);
            FontMetrics fm = g.getFontMetrics();
            g.drawString(label, LEFT - fm.stringWidth(label) - 8, y + 4);
        }
    }

    private static void drawXTicks(Graphics2D g, double min, double max) {
        int plotW = W - LEFT - RIGHT;
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        g.setColor(Color.DARK_GRAY);
        for (int i = 0; i <= 5; i++) {
            double v = min + (max - min) * i / 5.0;
            int x = LEFT + (plotW * i) / 5;
            String label = fmt(v);
            FontMetrics fm = g.getFontMetrics();
            g.drawString(label, x - fm.stringWidth(label) / 2, H - BOTTOM + 24);
        }
    }

    private static void drawRotatedLabel(Graphics2D g, String label, int x, int y) {
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
        g.setColor(Color.DARK_GRAY);
        Graphics2D copy = (Graphics2D) g.create();
        copy.rotate(-Math.PI / 4, x, y);
        String shortLabel = label.length() > 32 ? label.substring(0, 29) + "..." : label;
        FontMetrics fm = copy.getFontMetrics();
        copy.drawString(shortLabel, x - fm.stringWidth(shortLabel) / 2, y);
        copy.dispose();
    }

    private static void drawLegend(Graphics2D g, String[] names, int x, int y) {
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 13));
        for (int i = 0; i < names.length; i++) {
            g.setColor(PALETTE[i % PALETTE.length]);
            g.fillRect(x, y + i * 20, 14, 14);
            g.setColor(Color.DARK_GRAY);
            g.drawRect(x, y + i * 20, 14, 14);
            g.drawString(names[i], x + 20, y + i * 20 + 12);
        }
    }

    private static double max(double[] values) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : values) m = Math.max(m, v);
        return m;
    }

    private static double min(double[] values) {
        double m = Double.POSITIVE_INFINITY;
        for (double v : values) m = Math.min(m, v);
        return m;
    }

    private static String fmt(double v) {
        double av = Math.abs(v);
        if (av >= 100000.0 || (av > 0.0 && av < 0.001)) return String.format(Locale.US, "%.2e", v);
        if (av >= 100.0) return String.format(Locale.US, "%.1f", v);
        if (av >= 10.0) return String.format(Locale.US, "%.2f", v);
        return String.format(Locale.US, "%.4f", v);
    }
}
