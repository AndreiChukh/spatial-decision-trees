package sdt;

import sdt.check.Chapter5Check;
import sdt.check.SatelliteSdtCheck;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public final class Main {
    private Main() {}

    public static void main(String[] args) throws IOException {
        System.setProperty("java.awt.headless", "true");
        boolean chapter5 = contains(args, "--chapter5");
        boolean allTrees = contains(args, "--all-trees");
        String[] cleaned = stripModeFlags(args);

        if (allTrees) {
            System.out.println("Running all implemented decision-tree experiments: Chapter 4 DT/SDT, then Chapter 5 LTDT/FTSDT.");
            SatelliteSdtCheck.run(cleaned);
            Chapter5Check.run(cleaned);
        } else if (chapter5) {
            Chapter5Check.run(cleaned);
        } else {
            SatelliteSdtCheck.run(cleaned);
        }
        System.exit(0);
    }

    private static boolean contains(String[] args, String flag) {
        for (String arg : args) if (flag.equals(arg)) return true;
        return false;
    }

    private static String[] stripModeFlags(String[] args) {
        List<String> out = new ArrayList<>();
        for (String arg : args) {
            if ("--chapter5".equals(arg) || "--all-trees".equals(arg)) continue;
            out.add(arg);
        }
        return out.toArray(String[]::new);
    }
}
