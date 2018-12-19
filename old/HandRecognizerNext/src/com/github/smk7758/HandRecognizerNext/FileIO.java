package com.github.smk7758.HandRecognizerNext;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;

public class FileIO {
	public static void exportListPoint(Path file_path, List<Point> list) {
		// Path points_sorted_path = Paths.get("C:\\Users\\kariyassh\\Desktop\\2018-08-03_points_sorted.log");
		createFileWhenNotExists(file_path);
		try (BufferedWriter br = Files.newBufferedWriter(file_path)) {
			for (Point point : list) {
				br.write(point.x + ", " + point.y + "\r\n");
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	public static void exportList(Path file_path, List<double[]> list) {
		// Path points_sorted_path = Paths.get("C:\\Users\\kariyassh\\Desktop\\2018-08-03_points_sorted.log");
		createFileWhenNotExists(file_path);
		try (BufferedWriter br = Files.newBufferedWriter(file_path)) {
			for (double[] x : list) {
				if (x.length > 2) br.write(x[0] + ", " + x[1] + ", " + x[2] + "\r\n");
				else if (x.length > 1) br.write(x[0] + ", " + x[1] + "\r\n");
				else if (x.length > 0) br.write(x[0] + "\r\n");
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	public static void exportPoints(Path file_path, MatOfPoint points) {
		// Path points_sorted_path = Paths.get("C:\\Users\\kariyassh\\Desktop\\2018-08-03_points_sorted.log");
		createFileWhenNotExists(file_path);
		try (BufferedWriter br = Files.newBufferedWriter(file_path)) {
			for (Point point : points.toList()) {
				br.write(point.x + ", " + point.y + "\r\n");
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	private static void createFileWhenNotExists(Path file_path) {
		try {
			if (!Files.exists(file_path)) Files.createFile(file_path);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	public static String getFilePath(String path) {
		return getFilePath(path, "");
	}

	public static String getFilePath(String path, String add_text) {
		return getFilePath(path, add_text, path.substring(getLastDotIndex(path), path.length()));
	}

	public static String getFilePath(String path, String add_text, String extention) {
		if (add_text.length() > 0) add_text = "_" + add_text;
		return path.substring(0, getLastDotIndex(path)) + add_text + "_" + System.currentTimeMillis() + "." + extention;
	}

	public static int getLastDotIndex(String path) {
		int last_index = 0;
		for (int i = 0; i < path.length(); i++) {
			if (path.charAt(i) == '.') last_index = i;
		}
		return last_index;
	}
}
