package com.github.smk7758.MouseControle;

import java.awt.Point;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class FileIO {
	public static List<Point> getPoints(Path path) {
		List<Point> points = new ArrayList<>();
		try (Stream<String> lines = Files.lines(path)) {
			lines.forEachOrdered(line -> {
				String[] elements = line.split(",");
				points.add(new Point(Double.valueOf(elements[0].trim()).intValue(),
						Double.valueOf(elements[1].trim()).intValue()));
			});
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		return points;
	}
}
