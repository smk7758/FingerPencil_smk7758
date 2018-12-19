package com.github.smk7758.MouseControle;

import java.awt.AWTException;
import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.Robot;
import java.awt.event.InputEvent;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {
	final String filePath = "S:\\FingerPencil\\movie\\CIMG1780_perspected_points_1544884008082.txt";

	public static void main(String[] args) throws AWTException {
		// Robot robot = new Robot();
		// Dimension screen_size = Toolkit.getDefaultToolkit().getScreenSize();
		// robot.mouseMove(screen_size.width / 2, screen_size.height / 2);
		// robot.delay(1000);
		// robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
		// robot.delay(100);
		// robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
		new Main().shiftMouse();
	}

	public void shiftMouse() throws AWTException {
		try {
			Thread.sleep(5000);
		} catch (InterruptedException ex) {
			ex.printStackTrace();
		}
		System.out.println("ST");
		final Point firstMousePoint = MouseInfo.getPointerInfo().getLocation();

		Robot robot = new Robot();
		robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);

		Path path = Paths.get(filePath);
		List<Point> points = FileIO.getPoints(path);
		convertPoints(points).forEach(point -> {
			try {
				Thread.sleep(100);
			} catch (InterruptedException ex) {
				ex.printStackTrace();
			}
			robot.mouseMove(firstMousePoint.x + point.x, firstMousePoint.y + point.y);
		});
		robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);

		Point mousePoint_ = MouseInfo.getPointerInfo().getLocation();
		System.out.println(mousePoint_.x + ", " + mousePoint_.y);
	}

	public List<Point> convertPoints(List<Point> points) {
		List<Point> points_new = new ArrayList<>();
		for (Point point : points) {
			points_new.add(new Point(point.x / 6, point.y / 6));
		}
		return points_new;
	}
}
