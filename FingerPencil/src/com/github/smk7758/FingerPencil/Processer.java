package com.github.smk7758.FingerPencil;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
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
	private final String VIDEO_PATH;
	// OpenCV => 0<H<180, 0<S<255, 0<V<255
	// GIMP => 0<H<360, 0<S<100, 0<V<100
	final Scalar minHsvBlue = new Scalar(100, 20, 20), maxHsvBlue = new Scalar(135, 255, 255);
	final Scalar minHsvRed_0 = new Scalar(0, 90, 50), maxHsvRed_0 = new Scalar(10, 255, 255);
	final Scalar minHsvRed_1 = new Scalar(160, 90, 50), maxHsvRed_1 = new Scalar(179, 255, 255);
	// final Scalar hsv_blue_min = new Scalar(90, 50, 30), hsv_blue_max = new Scalar(135, 255, 200);
	// final Scalar hsv_red_min = new Scalar(150, 50, 50), hsv_red_max = new Scalar(180, 255, 200);
	// final Point left_point_up = new Point(567, 559), left_point_down = new Point(238, 894),
	// right_point_up = new Point(1353, 541), right_point_down = new Point(1642, 877);
	Mat homographyMatrix = null;

	public Processer(String videoPath) {
		this.VIDEO_PATH = videoPath;
	}

	public void run() {
		VideoCapture vc = new VideoCapture();
		{
			// 入力動画の初期化
			vc.open(VIDEO_PATH);
		}
		VideoWriter vw = new VideoWriter();
		{
			// 出力動画ファイルの初期化
			// avi -> 'M', 'J', 'P', 'G'
			// mp4 -> 32
			vw.open(FileIO.getFilePath(VIDEO_PATH, "finger_points", "mp4"), 32, 29,
					new Size(vc.get(Videoio.CV_CAP_PROP_FRAME_WIDTH), vc.get(Videoio.CV_CAP_PROP_FRAME_HEIGHT)));
		}

		Mat mat = new Mat();
		// Mat homographyMatrix = null;
		Mat matFirst = null;
		List<Point> fingerPoints = new ArrayList<>();
		List<double[]> perspectedPoints = new ArrayList<>();
		while (vc.isOpened() && vc.read(mat) && mat != null && !mat.empty()) {
			if (matFirst == null) matFirst = mat.clone();
			boolean test = loop(vc, mat, matFirst, fingerPoints, perspectedPoints, vw);
			if (!test) break;
		}

		{
			final Mat outputFingerPointsMat = matFirst.clone();
			fingerPoints
					.forEach(point -> Imgproc.circle(outputFingerPointsMat, point, 3, new Scalar(0, 255, 0), -1,
							Imgproc.LINE_8));

			Imgcodecs.imwrite(FileIO.getFilePath(VIDEO_PATH, "finger_points", "jpg"), outputFingerPointsMat);
		}
		{
			final Mat outputPerspectedMat = Mat.zeros(new Size(600, 600), CvType.CV_16S);
			perspectedPoints
					.forEach(point -> Imgproc.circle(outputPerspectedMat, new Point(point), 1, new Scalar(255), -1,
							Imgproc.LINE_8));

			Imgcodecs.imwrite(FileIO.getFilePath(VIDEO_PATH, "perspected_points", "jpg"), outputPerspectedMat);
		}
		{
			FileIO.exportList(Paths.get(FileIO.getFilePath(VIDEO_PATH, "perspected_points", "txt")), perspectedPoints);
		}

		vc.release();
		vw.release();

		System.out.println("FINISH!!");
	}

	// false -> 続行不能。
	// true -> 続行可能。
	public boolean loop(VideoCapture vc, Mat mat, Mat firstMat, List<Point> fingerPoints,
			List<double[]> perspectedPoints, VideoWriter vw) {
		Mat matHsv = new Mat();
		Mat srcPerspectMat = null;
		Mat perspectedMat = null;
		Mat outputMat = mat.clone();

		ListMap<Point, Integer> detectedMarkerPoints = detectMarkerPoints(mat, outputMat);
		if (detectedMarkerPoints.size() < 4) {
			System.err.println("Cannot detect all of markers.");
		}

		// TODO
		// {
		// for (Entry<Point, Integer> point : detectedMarkerPoints.entrySet()) {
		// System.out.println("Point: (" + point.getKey().x + ", " + point.getKey().y + "), ID: "
		// + point.getValue());
		// }
		// }

		Optional<Mat> homographyMatrix_tmp = getHomographyMatrix(detectedMarkerPoints);
		if (homographyMatrix_tmp.isPresent()) {
			homographyMatrix = homographyMatrix_tmp.get();
		} else {
			System.err.println("Cannot get homography matrix.");
		}

		if (homographyMatrix == null || homographyMatrix.empty()) {
			System.err.println("There is no homographyMatrix. -> Skip.");
			return true;
		}

		// TODO

		Imgproc.cvtColor(mat, matHsv, Imgproc.COLOR_BGR2HSV); // convert BGR -> HSV

		// 指の取得
		// List<MatOfPoint> contours = new ArrayList<>(); // 初期化
		//
		// Mat mat_hsv_diff = new Mat();
		//
		// Core.absdiff(firstMat, mat, mat_hsv_diff); // 差分を取る
		//
		// getSkinPart(contours, mat_hsv_diff);
		//
		// // 境界線を点としてとる
		//
		// // 最大面積をみつける
		// final int contour_max_area = getLargestArea(contours);
		//
		// if (contour_max_area < 0) {
		// System.err.println("Cannot find finger point in countour_max_area.");
		// return false;
		// }
		//
		// final MatOfPoint points = contours.get(contour_max_area);
		//
		// // ConvexHull
		// MatOfInt hull = new MatOfInt();
		// Imgproc.convexHull(points, hull, true);
		//
		// int[] hull_array = hull.toArray();
		// // drawConvexHullPoints(points, hull, mat);
		//
		// // 傾きの取得
		// System.out.println("最小の傾きの点の取得");
		// int smallest = getSmallestInclinationNumber(points, hull_array);
		// final Point fingerPoint = new Point(points.get(hull_array[smallest], 0));
		// fingerPoints.add(fingerPoint);
		//
		// Imgproc.circle(mat, fingerPoint, 30, new Scalar(0, 255, 0), -1, 4, 0);
		// 指の点の取得 (ここまでで)

		// 指取得(代替)
		final Optional<Point> fingerPoint = getSubstituteFingerPoint(matHsv, fingerPoints);

		if (!fingerPoint.isPresent()) {
			System.err.println("Cannot get finger point.");
			return true;
		}

		fingerPoints.add(fingerPoint.get());

		Imgproc.circle(outputMat, fingerPoint.get(), 5, new Scalar(0, 255, 0), -1, Imgproc.LINE_8);

		// 透視変換

		// 透視変換が行われる画像(src)の生成
		srcPerspectMat = Mat.zeros(firstMat.size(), CvType.CV_16SC1);

		// 透視変換が行われる画像に点を加える
		Imgproc.circle(srcPerspectMat, fingerPoint.get(), 2, new Scalar(255), -1, 4, 0);

		// 透視変換の結果を出力する画像の生成
		perspectedMat = new Mat(srcPerspectMat.size(), CvType.CV_16SC1);

		// 透視変換の実行
		Imgproc.warpPerspective(srcPerspectMat, perspectedMat, homographyMatrix, perspectedMat.size(),
				Imgproc.INTER_LINEAR);

		// トリミング処理
		final Rect rect = new Rect(5, 5, 600, 600);
		perspectedMat = new Mat(perspectedMat, rect);

		// 回転処理
		// final Mat matrix_ = Imgproc.getRotationMatrix2D(new Point(300, 300), -90.0, 1.0);
		// Imgproc.warpAffine(perspectedMat, perspectedMat, matrix_, new Size(600, 600));

		perspectedMat.convertTo(perspectedMat, CvType.CV_8UC1);

		Optional<List<double[]>> perspectedPoints_tmp = getCenterPointContrus(perspectedMat);
		if (!perspectedPoints_tmp.isPresent()) {
			System.err.println("Cannot get center point contrus.");
			return true;
		}

		final double[] perspectedPoint = perspectedPoints_tmp.get().get(0);
		perspectedPoints.add(perspectedPoint);

		vw.write(outputMat);
		return true;
	}

	public Optional<Point> getSubstituteFingerPoint(Mat matHsv, List<Point> fingerPoints) {

		// 指の点の取得 (ここから)
		Mat matHsvBlue = getBluePart(matHsv);
		Mat matHsvRed = getRedPart(matHsv);

		// getFingerPoint
		final Optional<List<double[]>> bluePoint = getCenterPointContrus(matHsvBlue),
				redPoint = getCenterPointContrus(matHsvRed);

		if (!bluePoint.isPresent() || !redPoint.isPresent()) {
			System.err.println("BluePoint or RedPoint is null (Cannot find the point).");
			return Optional.empty();
		} else if (bluePoint.get().size() < 1 || redPoint.get().size() < 1) {
			System.err.println("BluePoint or RedPoint is none (Cannot find the point).");
			return Optional.empty();
		}

		// 扱いやすくするためにこうした。
		final double[] bluePoint_ = bluePoint.get().get(0),
				redPoint_ = redPoint.get().get(0);

		final double fingerDiff_y = bluePoint_[1] - redPoint_[1];
		return Optional.of(new Point(redPoint_[0], redPoint_[1] - (fingerDiff_y / 5)));
	}

	public Mat getRedPart(Mat matHsv) {
		Mat matRedPart0 = matHsv.clone();
		Mat matRedPart1 = matHsv.clone();

		Core.inRange(matRedPart0, minHsvRed_0, maxHsvRed_0, matRedPart0);
		Core.inRange(matRedPart1, minHsvRed_1, maxHsvRed_1, matRedPart1);

		Core.add(matRedPart0, matRedPart1, matRedPart0);
		return matRedPart0;
	}

	public Mat getBluePart(Mat matHsv) {
		Mat matBluePart = matHsv.clone();
		Core.inRange(matBluePart, minHsvBlue, maxHsvBlue, matBluePart);
		return matBluePart;
	}

	public Optional<List<double[]>> getCenterPointContrus(Mat mat) {
		List<double[]> result = new ArrayList<>();
		List<MatOfPoint> contours = new ArrayList<>(); // 境界線の集合

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
			// 重心
			result.add(getCenter(contour_points));
		}
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

	/**
	 * 大きい順にソートして、面積も共に返す。
	 *
	 * @param contours 境界線の集合
	 * @return 大きい順にソートして、contoursでのIndexと、その境界線の面積。
	 */
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

	private Mat getSkinPart(List<MatOfPoint> contours, Mat mat_hsv) {
		Mat mat_skin = new Mat();
		final Scalar hsv_skin_min = new Scalar(0, 30, 60), hsv_skin_max = new Scalar(20, 150, 255);
		Core.inRange(mat_hsv, hsv_skin_min, hsv_skin_max, mat_skin); // 色範囲選択

		Imgproc.findContours(mat_skin, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_L1);

		return mat_skin;
	}

	@Override
	public void close() throws IOException {

	}

	/**
	 * 台形4点の取得ができていないと実行できない。 透視変換行列の取得
	 *
	 * @return 透視変換行列, ただしMarkerを取得出来なかった時、nullを返す。
	 */
	public Optional<Mat> getHomographyMatrix(ListMap<Point, Integer> detectMarkerPoints) {
		// 透視変換基底の前(変換前の基準4点)

		final double[] srcPoint = new double[8];
		for (int i = 0; i < 4; i++) {
			if (!detectMarkerPoints.containsValue(i)) {
				System.out.println(i);
				return Optional.empty(); // 以前のやつを使う。
			}
			srcPoint[2 * i] = detectMarkerPoints.getKey(i).x;
			srcPoint[2 * i + 1] = detectMarkerPoints.getKey(i).y;
		}

		Mat srcPointMat = new Mat(4, 2, CvType.CV_32F);
		srcPointMat.put(0, 0, srcPoint);

		// 透視変換基底の後(変換後の基準4点の生成)
		final double[] dstPoint = { 5, 5, 600, 5, 5, 600, 600, 600 }; // UP_Left, UP_Right, DOWN_Left, DOWN_Right
		Mat dstPointMat = new Mat(4, 2, CvType.CV_32F);
		dstPointMat.put(0, 0, dstPoint);

		// 透視変換基底行列の取得
		Mat mapMatrix = Imgproc.getPerspectiveTransform(srcPointMat, dstPointMat);

		// TODO
		if (mapMatrix == null) System.err.println("matMatrix is null @getHomographyMatrix");

		return Optional.ofNullable(mapMatrix);
	}

	public ListMap<Point, Integer> detectMarkerPoints(Mat inputMat, Mat outputMat) {
		// 台形4点のマーカーの中心座標。

		final Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_4X4_50);

		List<Mat> corners = new ArrayList<>();
		Mat markerIds = new Mat();
		// DetectorParameters parameters = DetectorParameters.create();
		Aruco.detectMarkers(inputMat, dictionary, corners, markerIds);

		if (outputMat != null) {
			Aruco.drawDetectedMarkers(outputMat, corners);
		}

		ListMap<Point, Integer> detectedMarkerPoints = new ListMap<>();

		final List<Point> markerCenterPoints = getCornerCenterPoints(corners);
		// markerIds.get(i, 0);
		// cornerPoints.get(i):

		markerCenterPoints
				.forEach(point -> Imgproc.circle(outputMat, point, 5, new Scalar(0, 0, 255), -1, Imgproc.LINE_8));

		for (int i = 0; i < markerCenterPoints.size() && i < markerIds.rows(); i++) {
			detectedMarkerPoints.add(markerCenterPoints.get(i), (int) markerIds.get(i, 0)[0]);
		}
		return detectedMarkerPoints;
	}

	/**
	 * マーカーのコーナーの中央の点の取得
	 *
	 * @param corners Arucoで取得された点
	 * @return マーカーのコーナーの中央の点
	 */
	public List<Point> getCornerCenterPoints(List<Mat> corners) {
		List<Point> points = new ArrayList<>();
		for (Mat mat : corners) {
			// TODO
			// System.out.println(mat.dump());

			List<Point> cornerPoints = new ArrayList<>();
			for (int row = 0; row < mat.height(); row++) {
				for (int col = 0; col < mat.width(); col++) {
					cornerPoints.add(new Point(mat.get(row, col)));
				}
			}

			points.add(new Point(getCenter(cornerPoints)));
		}
		return points;
	}

	public double[] getCenter(List<Point> points) {
		final MatOfPoint points_ = new MatOfPoint();
		points_.fromList(points);
		return getCenter(points_);
	}

	public double[] getCenter(MatOfPoint points) {
		// 重心を取得(投げてるver)
		Moments moments = Imgproc.moments(points);
		double[] center = { moments.get_m10() / moments.get_m00(), moments.get_m01() / moments.get_m00() };
		return center;
	}

}
