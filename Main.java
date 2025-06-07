import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        // Prompt user for dataset filename
        Scanner scanner = new Scanner(System.in);
        System.out.println("Welcome to Bertie Woosters Feature Selection Algorithm.");
        System.out.print("Type in the name of the file to test: ");
        String filename = scanner.nextLine().trim();

        // Read all data rows into a list of double arrays
        List<double[]> data = new ArrayList<>();
        int numFeatures = 0;

        // Load the file line by line
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;  // skip blank lines

                // Split on whitespace
                String[] parts = line.split("\\s+");
                if (parts.length < 2) {
                    System.out.println("Skipping bad line: " + line);
                    continue;
                }

                // Parse the tokens into a double array
                double[] instance = new double[parts.length];
                try {
                    for (int i = 0; i < parts.length; i++) {
                        instance[i] = Double.parseDouble(parts[i]);
                    }
                } catch (NumberFormatException e) {
                    System.out.println("Bad number in line: " + line);
                    continue;
                }

                // First element is the class label; verify it is 1 or 2
                if ((int) instance[0] != 1 && (int) instance[0] != 2) {
                    System.out.println("Invalid label in line: " + line);
                    continue;
                }

                data.add(instance);
                numFeatures = parts.length - 1;  // subtract label column
            }
            br.close();
        } catch (IOException e) {
            System.out.println("File error: " + e.getMessage());
            scanner.close();
            return;
        }

        // If no data was loaded, exit
        if (data.isEmpty()) {
            System.out.println("No valid data found.");
            scanner.close();
            return;
        }

        // Report dataset size and number of features (excluding class label)
        System.out.println("This dataset has " + numFeatures +
                           " features (not including the class attribute), with " +
                           data.size() + " instances.\n");

        // Normalize each feature column to mean 0 and standard deviation 1
        normalize(data, numFeatures);

        // Compute and show baseline accuracy using all features
        List<Integer> allFeatures = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) allFeatures.add(i);
        double baseline = calculateAccuracy(data, allFeatures);
        System.out.printf(
            "Running nearest neighbor with all %d features, using \"leaving-one-out\" evaluation, " +
            "I get an accuracy of %.1f%%%n%n", numFeatures, baseline);

        // Ask user which algorithm to run: Forward Selection or Backward Elimination
        System.out.println("Type the number of the algorithm you want to run.\n");
        System.out.println("1) Forward Selection");
        System.out.println("2) Backward Elimination");
        int choice = scanner.nextInt();
        scanner.close();
        System.out.println();

        if (choice == 1) {
            System.out.println("Beginning search.\n");
            long startTime = System.nanoTime();
            forwardSelection(data, numFeatures);
            long endTime = System.nanoTime();
            double elapsedSec = (endTime - startTime) / 1e9;
            System.out.printf("%nTime taken: %.4f seconds%n", elapsedSec);
        } else if (choice == 2) {
            System.out.println("Beginning search.\n");
            long startTime = System.nanoTime();
            backwardElimination(data, numFeatures);
            long endTime = System.nanoTime();
            double elapsedSec = (endTime - startTime) / 1e9;
            System.out.printf("%nTime taken: %.4f seconds%n", elapsedSec);
        } else {
            System.out.println("Bad choice, exiting.");
        }
    }

    // Normalize features to mean 0 and standard deviation 1
    static void normalize(List<double[]> data, int numFeatures) {
        for (int f = 1; f <= numFeatures; f++) {
            // Compute mean of column f
            double sum = 0;
            for (double[] instance : data) {
                sum += instance[f];
            }
            double mean = sum / data.size();

            // Compute standard deviation of column f
            double sumSq = 0;
            for (double[] instance : data) {
                sumSq += (instance[f] - mean) * (instance[f] - mean);
            }
            double std = Math.sqrt(sumSq / data.size());

            // Normalize values in column f, skipping if std is zero
            if (std != 0) {
                for (double[] instance : data) {
                    instance[f] = (instance[f] - mean) / std;
                }
            }
        }
    }

    // Forward selection: add features one by one
    static void forwardSelection(List<double[]> data, int numFeatures) {
        List<Integer> features = new ArrayList<>();          // current set
        List<Integer> bestOverallSet = new ArrayList<>();    // best overall subset
        double bestOverallAcc = -1;

        double bestAccuracy = defaultAccuracy(data);  // accuracy with no features

        // First round: evaluate each single feature individually
        for (int f = 0; f < numFeatures; f++) {
            List<Integer> temp = new ArrayList<>();
            temp.add(f);
            double acc = calculateAccuracy(data, temp);
            System.out.printf("Using feature(s) %s accuracy is %.1f%%%n", formatFeatureSet(temp), acc);
            if (acc > bestAccuracy) {
                bestAccuracy = acc;
                features.clear();
                features.add(f);
            }
            if (acc > bestOverallAcc) {
                bestOverallAcc = acc;
                bestOverallSet = new ArrayList<>(temp);
            }
        }
        System.out.printf("%nFeature set %s was best, accuracy is %.1f%%%n%n",
                          formatFeatureSet(features), bestAccuracy);

        // Subsequent rounds: keep adding the best next feature 
        while (features.size() < numFeatures) {
            int bestFeature = -1;
            double thisRoundBest = -1;

            // Try adding each candidate feature not already in the set
            for (int f = 0; f < numFeatures; f++) {
                if (features.contains(f)) continue;
                List<Integer> temp = new ArrayList<>(features);
                temp.add(f);
                double acc = calculateAccuracy(data, temp);
                System.out.printf("Using feature(s) %s accuracy is %.1f%%%n", formatFeatureSet(temp), acc);
                if (acc > thisRoundBest) {
                    thisRoundBest = acc;
                    bestFeature = f;
                }
                if (acc > bestOverallAcc) {
                    bestOverallAcc = acc;
                    bestOverallSet = new ArrayList<>(temp);
                }
            }

            // If no feature improves, break (should not happen since thisRoundBest starts at -1)
            if (bestFeature == -1) {
                break;
            }
            if (thisRoundBest < bestAccuracy) {
                System.out.println("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n");
            }
            features.add(bestFeature);
            bestAccuracy = thisRoundBest;
            System.out.printf("Feature set %s was best, accuracy is %.1f%%%n%n",
                              formatFeatureSet(features), bestAccuracy);
        }

        // Final report of the best overall subset found
        System.out.println("Finished search!! The best feature subset is " +
                           formatFeatureSet(bestOverallSet) + ", which has an accuracy of " +
                           String.format("%.1f%%.", bestOverallAcc));
    }

    // Backward elimination: remove features one by one
    static void backwardElimination(List<double[]> data, int numFeatures) {
        List<Integer> features = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) features.add(i);
        List<Integer> bestOverallSet = new ArrayList<>(features);
        double bestOverallAcc = calculateAccuracy(data, features);

        double bestAccuracy = bestOverallAcc;

        // First round: try removing each feature individually
        for (int f : new ArrayList<>(features)) {
            List<Integer> temp = new ArrayList<>(features);
            temp.remove(Integer.valueOf(f));
            double acc = calculateAccuracy(data, temp);
            System.out.printf("Using feature(s) %s accuracy is %.1f%%%n", formatFeatureSet(temp), acc);
            if (acc > bestAccuracy) {
                bestAccuracy = acc;
                features.remove(Integer.valueOf(f));
            }
            if (acc > bestOverallAcc) {
                bestOverallAcc = acc;
                bestOverallSet = new ArrayList<>(temp);
            }
        }
        System.out.printf("%nFeature set %s was best, accuracy is %.1f%%%n%n",
                          formatFeatureSet(features), bestAccuracy);

        // Subsequent rounds: keep removing the worst feature even if accuracy drops
        while (features.size() > 1) {
            int worstFeature = -1;
            double thisRoundBest = -1;

            for (int f : new ArrayList<>(features)) {
                List<Integer> temp = new ArrayList<>(features);
                temp.remove(Integer.valueOf(f));
                double acc = calculateAccuracy(data, temp);
                System.out.printf("Using feature(s) %s accuracy is %.1f%%%n", formatFeatureSet(temp), acc);
                if (acc > thisRoundBest) {
                    thisRoundBest = acc;
                    worstFeature = f;
                }
                if (acc > bestOverallAcc) {
                    bestOverallAcc = acc;
                    bestOverallSet = new ArrayList<>(temp);
                }
            }

            // If no removal improves, break (should not happen since thisRoundBest starts at -1)
            if (worstFeature == -1) {
                break;
            }
            if (thisRoundBest < bestAccuracy) {
                System.out.println("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n");
            }
            features.remove(Integer.valueOf(worstFeature));
            bestAccuracy = thisRoundBest;
            System.out.printf("Feature set %s was best, accuracy is %.1f%%%n%n",
                              formatFeatureSet(features), bestAccuracy);
        }

        // Final report of the best overall subset found
        System.out.println("Finished search!! The best feature subset is " +
                           formatFeatureSet(bestOverallSet) + ", which has an accuracy of " +
                           String.format("%.1f%%.", bestOverallAcc));
    }

    // Check accuracy with leave-one-out evaluation over selected features
    static double calculateAccuracy(List<double[]> data, List<Integer> features) {
        if (features.isEmpty()) return defaultAccuracy(data);

        int correct = 0;
        for (int i = 0; i < data.size(); i++) {
            double[] test = data.get(i);
            double minDist = Double.MAX_VALUE;
            int predicted = -1;
            for (int j = 0; j < data.size(); j++) {
                if (j == i) continue;
                double[] other = data.get(j);
                double dist = 0;
                // Compute Euclidean distance over selected features
                for (int f : features) {
                    double diff = test[f + 1] - other[f + 1];
                    dist += diff * diff;
                }
                dist = Math.sqrt(dist);
                if (dist < minDist) {
                    minDist = dist;
                    predicted = (int) other[0];
                }
            }
            if (predicted == (int) test[0]) correct++;
        }
        return 100.0 * correct / data.size();
    }

    // Default accuracy: guess the most common class label
    static double defaultAccuracy(List<double[]> data) {
        int count1 = 0, count2 = 0;
        for (double[] instance : data) {
            if ((int) instance[0] == 1) count1++;
            else count2++;
        }
        return 100.0 * Math.max(count1, count2) / data.size();
    }

    // Format a zero-based feature list as a 1-based brace-enclosed set
    // e.g. [1, 7] -> "{2, 8}"
    static String formatFeatureSet(List<Integer> features) {
        if (features.isEmpty()) return "{}";
        StringBuilder sb = new StringBuilder("{");
        for (int i = 0; i < features.size(); i++) {
            sb.append(features.get(i) + 1);
            if (i < features.size() - 1) sb.append(", ");
        }
        sb.append("}");
        return sb.toString();
    }
}
