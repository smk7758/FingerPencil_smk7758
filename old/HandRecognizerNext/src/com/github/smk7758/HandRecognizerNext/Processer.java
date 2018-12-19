package com.github.smk7758.HandRecognizerNext;

import java.io.Closeable;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

public class Processer implements Closeable {
	private final String video_path;
	private VideoCapture vc = null;
	private VideoWriter vw = null;
	private VideoWriter vwTest = null;
	final Point left_point_up = new Point(567, 559), left_point_down = new Point(238, 894),
			right_point_up = new Point(1353, 541), right_point_down = new Point(1642, 877);
	// final Scalar hsv_blue_min = new Scalar(100, 90, 80), hsv_blue_max = new Scalar(135, 255, 200);
	// final Scalar hsv_red_min = new Scalar(165, 90, 120), hsv_red_max = new Scalar(180, 255, 200);
	// private final Scalar hsv_green_min = new Scalar(90, 40, 50), hsv_green_max = new Scalar(100, 180, 180);

	public Processer(String video_path) {
		this.video_path = video_path;
		this.vc = new VideoCapture();
		this.vw = new VideoWriter();
		this.vwTest = new VideoWriter();
	}

	// @Override
	// public void run() {
	// if (!canDo()) {
	// System.err.println("Cannot start.");
	// return;
	// }
	// recognizer();
	// }

	// TODO
	public boolean canDo() {
		return true;
	}

	public void recognizer() {
		List<Point> finger_points = new ArrayList<>();
		List<double[]> pointsAfter = new ArrayList<>();

		// 入力動画の初期化
		vc.open(video_path);

		// 出力動画ファイルの初期化
		// avi -> 'M', 'J', 'P', 'G'
		// mp4 -> 32
		vw.open(FileIO.getFilePath(video_path, "finger_points", "mp4"), 32, 29,
				new Size(vc.get(Videoio.CV_CAP_PROP_FRAME_WIDTH), vc.get(Videoio.CV_CAP_PROP_FRAME_HEIGHT)));

		vwTest.open(FileIO.getFilePath(video_path, "test", "mp4"), 32, 29,
				new Size(vc.get(Videoio.CV_CAP_PROP_FRAME_WIDTH), vc.get(Videoio.CV_CAP_PROP_FRAME_HEIGHT)));

		// 前のフレームの方をなくしたいときに使う
		// for (int i = 0; i < 10; i++) {
		// vc.read(mat);
		// }

		Mat mat = new Mat();
		Mat matHsv = null;
		Mat matFirst = null;
		Mat mapMatrix = null;
		Mat srcMat = null;
		Mat perspectedMat = null;
		if (!vc.isOpened()) System.err.println("!?!?");
		while (vc.isOpened() && vc.read(mat) && mat != null && !mat.empty()) {

			// 一番最初のフレームの取得
			if (matFirst == null) matFirst = mat.clone();

			if (mapMatrix == null) {
				// 透視変換の基底行列の取得
				mapMatrix = getPerspectiveTramsformMatrixMat();
			}
			// mat_result_convexhull_points = mat.clone();

			matHsv = new Mat();
			Imgproc.cvtColor(mat, matHsv, Imgproc.COLOR_BGR2HSV); // convert BGR -> HSV
			// mat = exceptLightPrevention(mat, mat_hsv); // 光影響範囲の除外かつHSVで返す。

			// 指の取得
			List<MatOfPoint> contours = new ArrayList<>(); // 初期化

			Mat mat_hsv_diff = new Mat();

			Core.absdiff(matFirst, mat, mat_hsv_diff); // 差分を取る

			getSkinPart(contours, mat_hsv_diff);

			vwTest.write(mat_hsv_diff);

			// 境界線を点としてとる

			// 最大面積をみつける
			final int contour_max_area = getLargestArea(contours);

			if (contour_max_area < 0) {
				System.err.println("Cannot find finger point in countour_max_area.");
				continue;
			}

			final MatOfPoint points = contours.get(contour_max_area);

			// ConvexHull
			MatOfInt hull = new MatOfInt();
			Imgproc.convexHull(points, hull, true);

			int[] hull_array = hull.toArray();
			drawConvexHullPoints(points, hull, mat);

			// 傾きの取得
			System.out.println("最小の傾きの点の取得");
			int smallest = getSmallestInclinationNumber(points, hull_array);
			Point fingerPoint = new Point(points.get(hull_array[smallest], 0));
			finger_points.add(fingerPoint);

			Imgproc.circle(mat, fingerPoint, 30, new Scalar(0, 255, 0), -1, 4, 0);

			// 指の点の取得 (ここまでで)

			// 透視変換

			// 透視変換が行われる画像(src)の生成
			srcMat = Mat.zeros(matFirst.size(), CvType.CV_16SC1);

			// 透視変換が行われる画像に点を加える
			Imgproc.circle(srcMat, fingerPoint, 2, new Scalar(255), -1, 4, 0);

			// 透視変換の結果を出力する画像の生成
			perspectedMat = new Mat(srcMat.size(), CvType.CV_16SC1);

			// 透視変換の実行
			Imgproc.warpPerspective(srcMat, perspectedMat, mapMatrix, perspectedMat.size(), Imgproc.INTER_LINEAR);

			final Rect rect = new Rect(5, 5, 600, 600);
			perspectedMat = new Mat(perspectedMat, rect);

			final Mat matrix_ = Imgproc.getRotationMatrix2D(new Point(300, 300), -90.0, 1.0);
			Imgproc.warpAffine(perspectedMat, perspectedMat, matrix_, new Size(600, 600));

			perspectedMat.convertTo(perspectedMat, CvType.CV_8UC1);

			Optional<List<double[]>> perspectedPoints = getCenterPointContrus(perspectedMat);
			if (perspectedPoints.isPresent()) {
				final double[] perspectedPoint = perspectedPoints.get().get(0);
				pointsAfter.add(perspectedPoint);
			}
			vw.write(mat);
		}

		for (Point finger_point : finger_points) {
			Imgproc.circle(matFirst, finger_point, 30, new Scalar(0, 255, 0), -1, 4, 0);
		}

		Imgcodecs.imwrite(FileIO.getFilePath(video_path, "test", "png"), matFirst);

		// // 透視変換
		//
		// // 透視変換の基底行列の取得
		// Mat mapMatrix = getPerspectiveTramsformMatrixMat();
		//
		// // 透視変換が行われる画像(src)の生成
		// Mat srcMat = Mat.zeros(first_mat.size(), CvType.CV_16SC1);
		// // final Scalar src = new Scalar(255);
		// for (Point fingerPoint : finger_points) {
		// Imgproc.circle(srcMat, fingerPoint, 2, new Scalar(255), -1, 4, 0);
		// }
		//
		// // 透視変換の結果を出力する画像の生成
		// Mat perspectedMat = new Mat(srcMat.size(), CvType.CV_16SC1);
		//
		// // 透視変換の実行
		// Imgproc.warpPerspective(srcMat, perspectedMat, mapMatrix, perspectedMat.size(), Imgproc.INTER_LINEAR);
		// Imgcodecs.imwrite(FileIO.getFilePath(video_path, "perspective_A", "png"), srcMat);
		// Imgcodecs.imwrite(FileIO.getFilePath(video_path, "perspective_B", "png"), perspectedMat);

		// FileIO.exportListPoint(Paths.get(FileIO.getFilePath(video_path, "2018-10-03", "txt")), result_finger_points);

		System.out.println("PointAfterSize: " + pointsAfter.size());
		FileIO.exportList(Paths.get(FileIO.getFilePath(video_path, "pointsAfter", "txt")), pointsAfter);

		vw.release();
		vwTest.release();
		vc.release();
		System.out.println("FINISH!");
	}

	public void drawConvexHullPoints(MatOfPoint points, MatOfInt hull, Mat export) {
		// System.out.println("Size: " + hull.size().height);
		double[] hull_point_i = { 0, 0 }, hull_point_k = { 0, 0 };
		int[] hull_array = hull.toArray();
		for (int i = 0; i < hull_array.length - 1; i++) {
			hull_point_i = points.get(hull_array[i], 0);
			hull_point_k = points.get(hull_array[i + 1], 0);
			Imgproc.line(export,
					new Point(hull_point_i), new Point(hull_point_k), new Scalar(0, 0, 255), 3);
			// System.out.println(hull_point_i[0] + ", " + hull_point_i[1]);
		}
		// System.out.println(hull_point_k[0] + ", " + hull_point_k[1]);
	}

	private Mat getSkinPart(List<MatOfPoint> contours, Mat mat_hsv) {
		Mat mat_skin = new Mat();
		final Scalar hsv_skin_min = new Scalar(0, 30, 60), hsv_skin_max = new Scalar(20, 150, 255);
		Core.inRange(mat_hsv, hsv_skin_min, hsv_skin_max, mat_skin); // 色範囲選択

		Imgproc.findContours(mat_skin, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_L1);

		return mat_skin;
	}

	public int getSmallestInclinationNumber(final MatOfPoint points, int[] hull_array) {
		List<Double> inclination_hull_points = new ArrayList<>(); // このインデックスはhullの点のインデックスと対応する。(ex: 1 -> (1, 2))

		for (int i = 0; i < hull_array.length - 1; i++) {
			final double inclination = getInclination(points.get(hull_array[i], 0), points.get(hull_array[i + 1], 0));
			// System.out.println("Inclination: " + inclination);
			inclination_hull_points.add(inclination);
		}

		// 最小傾きの取得
		int smallest = 0;
		for (int i = 0; i < inclination_hull_points.size(); i++) {
			if (Math.abs(inclination_hull_points.get(smallest)) > Math.abs(inclination_hull_points.get(i))) {
				smallest = i;
			}
		}
		// System.out.println("Smallest: " + smallest + ", size: " + inclination_hull_points.size());
		return smallest;
	}

	public double getInclination(double[] first_point, double[] second_point) {
		return (second_point[1] - first_point[1]) / (second_point[0] - first_point[0]);
	}

	public int getLargestArea(List<MatOfPoint> contours) {
		double area_max = 0, area = 0;
		int contour_max_area = -1;
		for (int j = 0; j < contours.size(); j++) {
			// area = contours.get(j).size().area(); //コッチはその面積のため違う
			area = Imgproc.contourArea(contours.get(j)); // 境界点の面積
			if (area_max < area) {
				area_max = area;
				contour_max_area = j;
			}
		}
		if (contour_max_area < 0) System.err.println("Cannot find contours area.");
		return contour_max_area;
	}

	/**
	 * 台形4点の取得ができていないと実行できない。 透視変換行列の取得
	 *
	 * @return
	 */
	public Mat getPerspectiveTramsformMatrixMat() {
		// 透視変換基底の前(変換前の基準4点)
		final double[] srcPoint = { left_point_up.x, left_point_up.y, left_point_down.x, left_point_down.y,
				right_point_up.x, right_point_up.y, right_point_down.x, right_point_down.y };
		Mat srcPointMat = new Mat(4, 2, CvType.CV_32F);
		srcPointMat.put(0, 0, srcPoint);

		// 透視変換基底の後(変換後の基準4点の生成)
		final double[] dstPoint = { 5, 5, 5, 600, 600, 5, 600, 600 };
		Mat dstPointMat = new Mat(4, 2, CvType.CV_32F);
		dstPointMat.put(0, 0, dstPoint);

		// 透視変換基底行列の取得
		Mat mapMatrix = Imgproc.getPerspectiveTransform(srcPointMat, dstPointMat);
		return mapMatrix;
	}

	public void replacePoints(Mat srcMat, List<Point> points, Scalar scalar) {
		if (points.isEmpty()) throw new IllegalArgumentException("Points is empty.");

		for (Point point : points) {
			srcMat.put((int) point.y, (int) point.x, scalar.val[1]);
		}
	}

	public void replacePoint(Mat srcMat, Point point, Scalar scalar) {
		srcMat.put((int) point.y, (int) point.x, new Scalar(255).val[0]);
	}

	public Mat exceptPrevention(Mat mat_gray_src, Mat mat_src_except_gray) {
		Mat mat_result = mat_gray_src.clone();

		for (int h = 0; h < mat_result.size().height; h++) {
			for (int w = 0; w < mat_result.size().width; w++) {
				if (isLightPoint(mat_src_except_gray, h, w)) {
					removeMatPoint(mat_result, h, w, 0); // 光影響範囲の除外
				}
			}
		}
		return mat_result;
	}

	public boolean isLightPoint(Mat mat_tmp_light_prevention, int h, int w) {
		if (mat_tmp_light_prevention.get(h, w)[0] > 0) return true;
		else return false;
	}

	public void removeMatPoint(Mat mat_gray, int h, int w, int under) {
		mat_gray.put(h, w, under); // 光影響範囲の除外
	}

	// /**
	// * 光影響範囲の除去。
	// *
	// * @param mat_bgr
	// * @param mat_hsv
	// * @return
	// */
	// public Mat exceptLightPrevention(Mat mat_bgr, Mat mat_hsv) {
	// Mat mat_light_result = mat_bgr.clone();
	//
	// // 「光ったところ」は値を持つ
	// Mat mat_tmp_light_prevention = getLightArea(mat_bgr);
	//
	// for (int h = 0; h < mat_light_result.size().height; h++) {
	// for (int w = 0; w < mat_light_result.size().width; w++) {
	// if (isLightPoint(mat_tmp_light_prevention, h, w)) {
	// removeLightPoint(mat_light_result, h, w); // 光影響範囲の除外
	// }
	// }
	// }
	// return mat_light_result;
	// }

	// /**
	// * 光ったところが値を持つように画像を返す。
	// *
	// * @param mat_bgr
	// * @return
	// */
	// public Mat getLightArea(Mat mat_bgr) {
	// mat_tmp_light_prevention = mat_bgr.clone();
	//
	// final int threshold_low = 200, threshold_up = 255; // threshold range
	// Imgproc.cvtColor(mat_tmp_light_prevention, mat_tmp_light_prevention, Imgproc.COLOR_BGRA2GRAY, 4); // グレースケール化
	// Imgproc.threshold(mat_tmp_light_prevention, mat_tmp_light_prevention, threshold_low, threshold_up,
	// Imgproc.THRESH_BINARY_INV); // 光ったところの除去
	// Core.bitwise_not(mat_tmp_light_prevention, mat_tmp_light_prevention); // 色反転して光ったところを値を持つように。
	// return mat_tmp_light_prevention;
	// }

	// public boolean isLightPoint(Mat mat_tmp_light_prevention, int h, int w) {
	// if (mat_tmp_light_prevention.get(h, w)[0] > 0) return true;
	// else return false;
	// }

	// public void removeLightPoint(Mat mat_light_result, int h, int w) {
	// mat_light_result.put(h, w, hsv_skin_min.val[0], hsv_skin_min.val[1], hsv_skin_min.val[2]); // 光影響範囲の除外
	// }

	public int getLargestAreaNumber(List<MatOfPoint> contours) {
		// if (contours.size() < 1) throw new IllegalArgumentException("No countrus.");
		double area_max = 0, area = 0;
		int contour_max_area = -1;
		for (int j = 0; j < contours.size(); j++) {
			// area = contours.get(j).size().area(); //コッチはその面積のため違う ??
			area = Imgproc.contourArea(contours.get(j)); // 普通の面積
			if (area_max < area) {
				area_max = area;
				contour_max_area = j;
			}
		}
		// if (contour_max_area < 0) System.out.println("Cannot find contours area.");
		return contour_max_area;
	}

	public double[] getCenter(MatOfPoint points) {
		// 重心を取得(投げてるver)
		Moments moments = Imgproc.moments(points);
		double[] center = { moments.get_m10() / moments.get_m00(), moments.get_m01() / moments.get_m00() };
		return center;
	}

	public Optional<List<double[]>> getCenterPointContrus(Mat mat) {
		List<double[]> result = new ArrayList<>();
		List<MatOfPoint> contours = new ArrayList<>();

		// 境界線を点としてとる(輪郭線)
		Imgproc.findContours(mat, contours, new Mat(mat.size(), mat.type()), Imgproc.RETR_EXTERNAL,
				Imgproc.CHAIN_APPROX_NONE);

		if (contours.size() < 1) {
			System.out.println("Cannot find any point! : after findContours @getCenterPointContrus");
			return Optional.empty();
		}

		ListMap<Integer, Double> contours_area = getSortedAreaNumber(contours);

		if (contours_area.size() < 0) {
			System.err.println("Cannot find any point! : after getLargestAreaNumber @getCenterPointContrus");
			return Optional.empty();
		}

		for (int i = 0; i < contours_area.size(); i++) {
			MatOfPoint contour_points = contours.get(contours_area.get(i).getKey());
			result.add(getCenter(contour_points));
		}
		// 重心
		return Optional.ofNullable(result);
	}

	public List<MatOfPoint> getContrus(Mat mat) {
		List<MatOfPoint> result = new ArrayList<>();
		List<MatOfPoint> contours = new ArrayList<>();

		// 境界線を点としてとる(輪郭線)
		Imgproc.findContours(mat, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

		if (contours.size() < 1) {
			System.out.println("Cannot find any point! : after findContours");
			return null;
		}

		ListMap<Integer, Double> contours_area = getSortedAreaNumber(contours);

		if (contours_area.size() < 0) {
			System.err.println("Cannot find any point! : after getLargestAreaNumber");
			return null;
		}

		for (int i = 0; i < contours_area.size(); i++) {
			MatOfPoint contour_points = contours.get(contours_area.get(i).getKey());
			result.add(contour_points);
		}
		// 重心
		return result;
	}

	public void drawConvexHullPoints(MatOfInt hull, MatOfPoint points, Mat export) {
		// System.out.println("Size: " + hull.size().height);
		double[] hull_point_i = { 0, 0 }, hull_point_k = { 0, 0 };
		int[] hull_array = hull.toArray();
		for (int i = 0; i < hull_array.length - 1; i++) {
			hull_point_i = points.get(hull_array[i], 0);
			hull_point_k = points.get(hull_array[i + 1], 0);
			Imgproc.line(export,
					new Point(hull_point_i), new Point(hull_point_k), new Scalar(0, 0, 255), 3);
			// System.out.println(hull_point_i[0] + ", " + hull_point_i[1]);
		}
		// System.out.println(hull_point_k[0] + ", " + hull_point_k[1]);
	}

	public MatOfPoint getPointGreenPart(Mat mat) {
		List<MatOfPoint> contours = new ArrayList<>();

		// 境界線を点としてとる(輪郭線)
		Imgproc.findContours(mat, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

		if (contours.size() < 1) {
			System.out.println("Cannot find any point! : after findContours");
			return null;
		}

		ListMap<Integer, Double> contours_area = getSortedAreaNumber(contours);

		if (contours_area.size() < 0) {
			System.err.println("Cannot find any point! : after getLargestAreaNumber");
			return null;
		}

		MatOfPoint contour_points = contours.get(contours_area.get(0).getKey());
		return contour_points;
	}

	// public Mat getGreenPart(Mat mat_hsv) {
	// Mat mat_part = mat_hsv.clone();
	// Core.inRange(mat_part, hsv_green_min, hsv_green_max, mat_part);
	// return mat_part;
	// }

	// public Mat getRedPart(Mat mat_hsv) {
	// Mat mat_red_part = mat_hsv.clone();
	// Core.inRange(mat_red_part, hsv_red_min, hsv_red_max, mat_red_part);
	// return mat_red_part;
	// }

	// public Mat getBluePart(Mat mat_hsv) {
	// Mat mat_blue_part = mat_hsv.clone();
	// Core.inRange(mat_blue_part, hsv_blue_min, hsv_blue_max, mat_blue_part);
	// return mat_blue_part;
	// }

	// public int getLargestAreaNumber(List<MatOfPoint> contours) {
	// // if (contours.size() < 1) throw new IllegalArgumentException("No countrus.");
	// double area_max = 0, area = 0;
	// int contour_max_area = -1;
	// for (int j = 0; j < contours.size(); j++) {
	// // area = contours.get(j).size().area(); //コッチはその面積のため違う ??
	// area = Imgproc.contourArea(contours.get(j)); // 普通の面積
	// if (area_max < area) {
	// area_max = area;
	// contour_max_area = j;
	// }
	// }
	// // if (contour_max_area < 0) System.out.println("Cannot find contours area.");
	// return contour_max_area;
	// }

	public ListMap<Integer, Double> getSortedAreaNumber(List<MatOfPoint> contours) {
		// if (contours.size() < 1) throw new IllegalArgumentException("No countrus.");
		double area = 0;
		ListMap<Integer, Double> contour_area = new ListMap<>();
		for (int i = 0; i < contours.size(); i++) {
			// area = contours.get(j).size().area(); //コッチはその面積のため違う ??
			area = Imgproc.contourArea(contours.get(i)); // 普通の面積
			if (contour_area.size() < 1 || contour_area.getValue(0) < area) {
				contour_area.add(0, i, area);
			} else {
				contour_area.add(i, area);
			}
		}
		return contour_area;
	}

	// public void drawConvexHullPoints(MatOfInt hull, MatOfPoint points, Mat export) {
	// // System.out.println("Size: " + hull.size().height);
	// double[] hull_point_i = { 0, 0 }, hull_point_k = { 0, 0 };
	// int[] hull_array = hull.toArray();
	// for (int i = 0; i < hull_array.length - 1; i++) {
	// hull_point_i = points.get(hull_array[i], 0);
	// hull_point_k = points.get(hull_array[i + 1], 0);
	// Imgproc.line(export,
	// new Point(hull_point_i), new Point(hull_point_k), new Scalar(0, 0, 255), 3);
	// System.out.println(hull_point_i[0] + ", " + hull_point_i[1]);
	// }
	// System.out.println(hull_point_k[0] + ", " + hull_point_k[1]);
	// }

	// @Override
	// public void interrupt() {
	// close();
	// super.interrupt();
	// }

	/**
	 * @return distance between finger_root_point_average and center.
	 */
	public double getBetweenDistance(double[] point_first, double[] point_second) {
		return getBetweenDistance(point_first[0], point_first[1], point_second[0], point_second[1]);
	}

	/**
	 * @return distance between finger_root_point_average and center.
	 */
	public double getBetweenDistance(double a_x, double a_y, double b_x, double b_y) {
		return Math.sqrt(Math.pow(a_x - b_x, 2)
				+ Math.pow(a_y - b_y, 2));
	}

	@Override
	public void close() {
		if (vc != null && vc.isOpened()) vc.release();
		// if (vw != null && vw.isOpened()) vw.release();
		System.out.println("Close. (Shower)");
	}
}
