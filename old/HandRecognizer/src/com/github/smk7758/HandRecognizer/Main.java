package com.github.smk7758.HandRecognizer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

//バグ: 同じ点がずっと出力されることがある。
//改善: どっちが上か下か
//バグ: controusが想定通りでない。
//バグ: 指先, 付け根の認識。
//バグ: 指付け根平均を用いた、指先制限が動いてない。
//バグ: 想定通りに点が並べられない。(_NONEの場合)->重心からくるくる回して点をやるのがいいかな。
public class Main {
	private String pic_path = "S:\\OpenCV_Things\\IMG_2075.JPG";
	/**
	 * 任意の調査点の前後の調査点の角度を求める際に、調査点の間隔を決めるもの。
	 */
	final int interval = 100;
	/**
	 * 指の点を除く際に重心からこの値だけ倍して、それ以下の点を除去するためのもの。
	 */
	final double multiplie_amount = 1.25;
	/**
	 * 肌色の範囲。
	 */
	final Scalar hsv_min = new Scalar(0, 30, 60), hsv_max = new Scalar(20, 150, 255);
	/**
	 * 画像データ, tmp画像データ(HSVに変換される前), 出力される画像データ(読み込まれた画像, HSVに変換される前)
	 */
	Mat mat, mat_tmp, mat_theta_result, mat_tmp_light_prevention, mat_local_result, mat_local_sorted_result,
			mat_convexhull_points;
	// , mat_zero_theta_result, mat_zero_local_result, mat_zero_convexhull_result;
	/**
	 * 輪郭でわけられた境界線(点)の集合のList。
	 */
	List<MatOfPoint> contours = new ArrayList<>();
	/**
	 * 最大面積の輪郭境界線(点)の集合。, sorted.
	 */
	MatOfPoint points, points_sorted;
	/**
	 * ThetaTypeと点の座標を保持するEntryのList.
	 */
	ListMap<double[], ThetaType> points_theta_types;
	/**
	 * LocalTypeと点の座標を保持するEntryのList.
	 */
	ListMap<double[], LocalType> points_local_types, points_local_types_sorted;
	/**
	 * CovexHullで取得したpointsの順番の値の集合。
	 */
	MatOfInt hull;
	/**
	 * 重心の座標。
	 */
	double[] center;
	/**
	 * FINGER_ROOT座標群の平均。
	 */
	double[] finger_root_point_average;
	/**
	 * 距離(?)
	 */
	double distance;
	/**
	 * 指先および付け根の判定をする角度の最大および最小値。
	 */
	final int finger_max = 150, finger_root_min = 200;

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	enum ThetaType {
		FINGER, FINGER_ROOT, OTHER;
	}

	enum LocalType {
		Min, Max, Other;
	}

	public static void main(String[] args) {
		new Main().recognizer();
	}

	public void recognizer() {
		if (!Files.exists(Paths.get(pic_path))) {
			System.err.println("File do not exsits.");
			return;
		}

		mat = Imgcodecs.imread(pic_path);
		mat_tmp = mat.clone();
		mat_theta_result = mat.clone();
		// mat_zero_theta_result = Mat.zeros(mat.size(), CvType.CV_8U);
		// mat_zero_local_result = Mat.zeros(mat.size(), CvType.CV_8U);
		// mat_zero_convexhull_result = Mat.zeros(mat.size(), CvType.CV_8U);

		// mat_all_points = mat.clone();

		// Imgproc.threshold(mat, mat, 80, 255, Imgproc.THRESH_BINARY); // 2値化

		Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2HSV); // convert BGR -> HSV

		mat = exceptLightPrevention(mat, mat_tmp); // 光影響範囲の除外かつHSVで返す。

		// Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2HSV); // convert BGR -> HSV
		Core.inRange(mat, hsv_min, hsv_max, mat); // 色範囲選択(min<=mat<=max)

		// final Mat element = Mat.ones(3, 3, CvType.CV_8UC1); // // dilate処理に必要な行列, 追加 3×3の行列で要素はすべて1
		// Imgproc.dilate(mat, mat, element, new Point(-1, -1), 3); // 塗りつぶし(うめうめ)
		// final Mat gaussian_kernal = Imgproc.getGaussianKernel(1,1);

		// 平均化フィルター(ぼかし)
		Imgproc.blur(mat, mat, new Size(5, 5));

		// 境界線を点としてとる(輪郭線), CHAIN_APPROX_NONEの場合、順番通りでない。
		Imgproc.findContours(mat, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

		// 最大面積をみつける
		points = contours.get(getLargestAreaNumber(contours));

		// 参照型の可能性もあるけど、たぶん大丈夫。
		// MatOfPoint all_points = points;

		// 重心
		center = getCenter(points);

		// [鋭角・鈍角判断] 任意の調査点の前後の調査点の角度を求め、手の指先か、付け根かを判別し、代入。 , (まだここでは不正確)
		points_theta_types = getPointsAngle(points);

		// [LocalMaxMin判断]
		// sortしてからじゃなきゃいけない。
		points_local_types = getLocalMinimumMaximum(points);

		// [ConvexHull判断]
		hull = new MatOfInt();
		Imgproc.convexHull(points, hull, true);

		// Sort!!
		points_sorted = sortPoints(points, center);

		System.out.println("==== points ====");
		Path points_path = Paths.get("F:\\users\\smk7758\\Desktop\\2018-08-03_points.log");
		try {
			if (!Files.exists(points_path)) Files.createFile(points_path);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		try (BufferedWriter br = Files.newBufferedWriter(points_path, StandardOpenOption.CREATE_NEW)) {
			for (Point point : points.toList()) {
				br.write(point.x + ", " + point.y + "\r\n");
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}

		System.out.println("==== points_sorted ====");
		Path points_sorted_path = Paths.get("F:\\users\\smk7758\\Desktop\\2018-08-03_points_sorted.log");
		try {
			if (!Files.exists(points_sorted_path)) Files.createFile(points_sorted_path);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		try (BufferedWriter br = Files.newBufferedWriter(points_sorted_path, StandardOpenOption.CREATE_NEW)) {
			for (Point point : points_sorted.toList()) {
				br.write(point.x + ", " + point.y + "\r\n");
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}

		// [LocalMaxMin判断] 順番を整えてから行う。
		points_local_types_sorted = getLocalMinimumMaximum(points_sorted);

		// FINGERにおいて、前後で同じものを一つにする処理
		// points_types = convertFingerPointsTogether(points_types);

		// 付け根の平均
		finger_root_point_average = getRootAverage(points_theta_types);

		// 付け根の平均と重心の距離
		// Math.sqrt()
		distance = getBetweenDistance(finger_root_point_average, center);

		// 付け根と重心との距離*1.25を満たさない指先の点を除去
		points_theta_types = expectShortDistance(points_theta_types, center, distance);

		// 円描画
		Imgproc.circle(mat_theta_result, new Point(center[0], center[1]), 100, new Scalar(0, 0, 0), 3, 4, 0);

		mat_local_result = mat_theta_result.clone();
		mat_local_sorted_result = mat_theta_result.clone();
		mat_convexhull_points = mat_theta_result.clone();

		// 点描画(角度)
		System.out.println("--ThetaPoints--");
		drawThetaPoints(points_theta_types, mat_theta_result);

		// 点描画(極小・極大値)
		System.out.println("--LocalPoints--");
		drawLocalPoints(points_local_types, mat_local_result);

		System.out.println("--LocalPoints_sorted--");
		drawLocalPoints(points_local_types_sorted, mat_local_sorted_result);

		// 線描画(ConvexHull)
		System.out.println("--ConvexHullPoints--");
		drawConvexHullPoints(hull, points, mat_convexhull_points);

		// 画像出力
		Imgcodecs.imwrite(getFilePath(pic_path, "theta"), mat_theta_result);
		Imgcodecs.imwrite(getFilePath(pic_path, "local"), mat_local_result);
		Imgcodecs.imwrite(getFilePath(pic_path, "local_sorted"), mat_local_sorted_result);
		Imgcodecs.imwrite(getFilePath(pic_path, "convexhull"), mat_convexhull_points);

		// System.out.println("AllPointsSize: " + all_points.size().height);
		// for (int i = 0; i < all_points.size().height; i += 10) {
		// Imgproc.circle(mat_all_points, new Point(all_points.get(i, 0)), 10, new Scalar(255, 0, 0), 2, 4, 0);
		// }
		// Imgcodecs.imwrite(getFilePath(pic_path, "all_points"), mat_all_points);
		//
		// // Imgcodecs.imwrite(getFilePath(pic_path, "light"), mat_tmp_light_prevention);

		// getDifferentiateValue(points).forEach(point -> System.out.println(point[0] + ", " + point[1]));
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
			System.out.println(hull_point_i[0] + ", " + hull_point_i[1]);
		}
		System.out.println(hull_point_k[0] + ", " + hull_point_k[1]);
	}

	public void drawLocalPoints(ListMap<double[], LocalType> points_local_types, Mat export) {
		points_local_types.entrySet().forEach(point -> {
			Imgproc.circle(export, new Point(point.getKey()), 10, new Scalar(0, 0, 0), 3, 4, 0);
			if (point.getValue().equals(LocalType.Max)) {
				Imgproc.circle(export, new Point(point.getKey()), 20, new Scalar(0, 0, 255), 3, 4, 0);
			} else if (point.getValue().equals(LocalType.Min)) {
				Imgproc.circle(export, new Point(point.getKey()), 20, new Scalar(0, 255, 0), 3, 4, 0);
			} else {
				Imgproc.circle(export, new Point(point.getKey()), 10, new Scalar(255, 0, 0), 3, 4, 0);
			}
			System.out.println(point.getKey()[0] + ", " + point.getKey()[1]);
		});
	}

	public void drawThetaPoints(ListMap<double[], ThetaType> points_theta_types, Mat export) {
		points_theta_types.entrySet().forEach(point -> {
			Imgproc.circle(export, new Point(point.getKey()), 10, new Scalar(0, 0, 0), 3, 4, 0);
			if (point.getValue().equals(ThetaType.FINGER)) {
				Imgproc.circle(export, new Point(point.getKey()), 20, new Scalar(0, 0, 255), 3, 4, 0);
			} else if (point.getValue().equals(ThetaType.FINGER_ROOT)) {
				Imgproc.circle(export, new Point(point.getKey()), 20, new Scalar(0, 255, 0), 3, 4, 0);
			} else {
				Imgproc.circle(export, new Point(point.getKey()), 10, new Scalar(255, 0, 0), 3, 4, 0);
			}
			System.out.println(point.getKey()[0] + ", " + point.getKey()[1]);
		});
	}

	public ListMap<double[], LocalType> getLocalMinimumMaximum(MatOfPoint points) {
		ListMap<double[], LocalType> result_points = new ListMap<>();
		double[] a = { 0, 0 }, b = { 0, 0 }, c = { 0, 0 };
		for (int i = interval; i < points.size().height - interval; i += interval) {
			a = points.get(i - interval, 0);
			b = points.get(i, 0);
			c = points.get(i + interval, 0);
			if (b[1] >= a[1] && b[1] >= c[1]) {
				if (!result_points.containsKey(b)) {
					System.out.println("Max: " + i);
					result_points.add(b, LocalType.Max);
				}
			} else if (b[1] <= a[1] && b[1] <= c[1]) {
				if (!result_points.containsKey(b)) {
					System.out.println("Min: " + i);
					result_points.add(b, LocalType.Min);
				}
			} else {
				if (!result_points.containsKey(b)) result_points.add(b, LocalType.Other);
			}
		}
		return result_points;
	}

	public Mat exceptLightPrevention(Mat mat, Mat mat_tmp) {
		mat_tmp_light_prevention = mat_tmp.clone();

		Imgproc.cvtColor(mat_tmp_light_prevention, mat_tmp_light_prevention, Imgproc.COLOR_RGBA2GRAY, 4); // グレースケール化
		Imgproc.threshold(mat_tmp_light_prevention, mat_tmp_light_prevention, 200, 255, Imgproc.THRESH_BINARY_INV); // 光ったところの除去
		Core.bitwise_not(mat_tmp_light_prevention, mat_tmp_light_prevention); // 色反転
		// 「光ったところ」は値を持つ

		for (int h = 0; h < mat.size().height; h++) {
			for (int w = 0; w < mat.size().width; w++) {
				if (mat_tmp_light_prevention.get(h, w)[0] > 0) {
					mat.put(h, w, hsv_min.val[0], hsv_min.val[1], hsv_min.val[2]); // 光影響範囲の除外
				}
			}
		}
		return mat;
	}

	public int getLargestAreaNumber(List<MatOfPoint> contours) {
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

	// public MatOfPoint sortPointsOfX(MatOfPoint points) {
	// // TODO
	// List<Point> points_ = points.toList();
	// points_.sort(new Comparator<Point>() {
	// @Override
	// public int compare(Point a, Point b) {
	// return (int) (a.x - b.x);
	// }
	// });
	// MatOfPoint points_result = new MatOfPoint();
	// points_result.fromList(points_);
	// return points_result;
	// }

	public MatOfPoint sortPoints(MatOfPoint points, double[] center) {
		// TODO
		// List<Point> points_ = points.toList();
		TreeMap<Double, Point> points_ = new TreeMap<>();
		for (int i = 0; i < points.size().width; i++) {
			double[] c = { center[0], 0 }, a_ = { 0, 0 }, c_ = { 0, 0 };
			final double theta_a, theta_c, theta;
			subtractionPoint(points.get(i, 0), center, c, a_, c_);
			theta_a = convertAngle(Math.atan(a_[1] / a_[0]), a_[0]);
			theta_c = convertAngle(Math.atan(c_[1] / c_[0]), c_[0]);
			theta = getAngleABC(theta_a, theta_c);
			points_.put(theta, new Point(points.get(i, 0)));
			// if (getBetweenDistance(points.get(i, 0), points.get(i + 1, 0)) > (mat.width() / 10)) {
			// // pointの距離が画像の長さの10分の1ならば。
			// List<Point> points_tmp = points_.subList(0, i);
			// for (int j = 0; j <= i; j++) {
			// points_.remove(0);
			// }
			// points_.addAll(points_tmp);
			// }
		}
		points_.entrySet().forEach(entry -> System.out.println(
				entry.getKey() + ": " + entry.getValue().x + ", " + entry.getValue().y));
		MatOfPoint points_result = new MatOfPoint();
		points_result.fromList(new ArrayList<>(points_.values()));
		return points_result;
	}

	public ListMap<double[], ThetaType> getPointsAngle(MatOfPoint points) {
		// 任意の調査点の前後の調査点の角度を求める。
		/**
		 * ThetaTypeと点の座標を保持するEntryのList.
		 */
		ListMap<double[], ThetaType> points_types = new ListMap<>(); // keyは必ずしもpointsと一致するとは限らない。

		double[] a = { 0, 0 }, b = { 0, 0 }, c = { 0, 0 };
		/**
		 * bを原点としてみた時の座標。
		 */
		double[] a_ = { 0, 0 }, c_ = { 0, 0 };
		double theta_a = 0, theta_c = 0;
		for (int i = 0; i < points.size().height; i += interval) {
			// final int interval = (int) (points.size().height / 10); // TODO: 全体の数からの比率
			// TODO: points.size().widthの割合でintervalを見るべき。
			// MatOfPoint::getはx, yのdouble[]を返す。(ただし上下逆)
			if (a[0] == 0 && a[1] == 0) {
				a = points.get(i, 0);
				continue;
			}
			if (b[0] == 0 && b[1] == 0) {
				b = points.get(i, 0);
				continue;
			}
			c = points.get(i, 0);

			// 原点 -> b
			subtractionPoint(a, b, c, a_, c_);

			theta_a = convertAngle(Math.atan(a_[1] / a_[0]), a_[0]);
			theta_c = convertAngle(Math.atan(c_[1] / c_[0]), c_[0]);

			// 同じ座標がないときに追加をする。
			if (!points_types.containsKey(c)) {
				double theta = getAngleABC(theta_a, theta_c);
				// TODO
				System.out.println(
						// "ThetaA: " + Math.toDegrees(theta_a) + ", atan_A: " + Math.atan(a_[1] / a_[0])
						// + ", ThetaC: " + Math.toDegrees(theta_c) + ", atan_C: " + Math.atan(c_[1] / c_[0])
						"Theta: " + Math.toDegrees(theta)
								+ "ThetaType: " + getPointTypes(theta)
				// + ", a(" + a[0] + ", " + a[1] + ")"
				// + ", b(" + b[0] + ", " + b[1] + ")"
				// + ", c(" + c[0] + ", " + c[1] + ")"
				// + ", a_(" + a_[0] + ", " + a_[1] + ")"
				// + ", c_(" + c_[0] + ", " + c_[1] + ")"
				);
				points_types.add(c, getPointTypes(theta));
			}
			// リストを前に進める。
			a = b;
			b = c;
		}
		System.out.println("OriginalSize: " + points.size().height);
		System.out.println("ResultSize:" + points_types.size());
		return points_types;
	}

	public void subtractionPoint(double[] a, double[] b, double[] c, double[] a_, double[] c_) {
		a_[0] = (a[0] - b[0]);
		a_[1] = (a[1] - b[1]);
		c_[0] = (c[0] - b[0]);
		c_[1] = (c[1] - b[1]);
	}

	/**
	 * @return 角度のThetaType
	 */
	public ThetaType getPointTypes(double theta_abc) {
		theta_abc = Math.toDegrees(theta_abc);
		// System.out.println(theta_abc);
		if (theta_abc <= finger_max) {
			// System.out.println("FINGER");
			return ThetaType.FINGER;
		} else if (finger_root_min <= theta_abc) {
			return ThetaType.FINGER_ROOT;
		} else {
			return ThetaType.OTHER;
		}
	}

	/**
	 * @param theta_a angle(0<=x<=2π)
	 * @param theta_c angle(0<=x<=2π)
	 * @return theta_aとtheta_cの差を、0<=theta<=2πの範囲で返します。
	 */
	public double getAngleABC(double theta_a, double theta_c) {
		return Math.abs(theta_a - theta_c);
	}

	/**
	 * x | x !<br>
	 * -----------<br>
	 * x_| x_ $<br>
	 * x : x>=0, x_ : x<0, ! : theta>=0, $ : theta<0
	 *
	 * @param theta atanによって取得された角度。
	 * @param x 任意の原点を中心としたときの座標。
	 * @return 角度(0<=theta<=2π)を返す
	 */
	public double convertAngle(double theta, double x) {
		if (theta >= 0) {
			if (x >= 0) return theta;
			else return theta + Math.PI;
		} else {
			if (x >= 0) return theta + 2 * Math.PI;
			else return theta + Math.PI;
		}
	}

	public ListMap<double[], ThetaType> convertFingerPointsTogether(ListMap<double[], ThetaType> points_types) {
		// TODO
		List<Point> tmp; // 同じポイントたちの重心を取るためのもの
		MatOfPoint points;
		Entry<double[], ThetaType> point = null;
		for (int i = 0; i < points_types.size() - 1; i++) {
			tmp = new ArrayList<>();
			point = points_types.get(i);
			points_types.remove(i);
			tmp.add(new Point(point.getKey()));
			for (int j = 0; (i + j) < points_types.size()
					&& points_types.get(i + j).getValue().equals(point.getValue()); j++) {
				tmp.add(new Point(points_types.get(i + 1).getKey()));
				points_types.remove(i + 1);
			}
			points = new MatOfPoint();
			points.fromList(tmp);
			points_types.add(getCenter(points), point.getValue());
			// get center??
			// remove points and add center
		}
		return points_types;
	}

	public double[] getCenter(MatOfPoint points) {
		// 重心を取得(投げてるver)
		Moments moments = Imgproc.moments(points);
		double[] center = { moments.get_m10() / moments.get_m00(), moments.get_m01() / moments.get_m00() };
		return center;
	}

	public double[] getRootAverage(ListMap<double[], ThetaType> points_types) {
		Entry<Integer, double[]> sums = getFingerRootPointSum(points_types);
		int finger_root_point_amount = sums.getKey();
		double[] finger_root_point_sum = sums.getValue();

		// 付け根の平均
		double[] finger_root_point_average = { 0, 0 };
		finger_root_point_average[0] = finger_root_point_sum[0] / finger_root_point_amount;
		finger_root_point_average[1] = finger_root_point_sum[1] / finger_root_point_amount;
		return finger_root_point_average;
	}

	public Entry<Integer, double[]> getFingerRootPointSum(ListMap<double[], ThetaType> points_types) {
		// 総和取得
		int finger_root_point_amount = 0; // 点の個数
		double[] finger_root_point_sum = { 0, 0 }; // 点座標の総和
		for (Entry<double[], ThetaType> point_type : points_types.entrySet()) {
			if (point_type.getValue().equals(ThetaType.FINGER_ROOT)) {
				double[] finger_root_point = point_type.getKey();
				finger_root_point_sum[0] += finger_root_point[0];
				finger_root_point_sum[1] += finger_root_point[1];

				finger_root_point_amount++;
			}
		}
		return new AbstractMap.SimpleEntry<Integer, double[]>(finger_root_point_amount, finger_root_point_sum);
	}

	/**
	 * @return distance between finger_root_point_average and center.
	 */
	public double getBetweenDistance(double[] point_first, double[] point_second) {
		return Math.sqrt(Math.pow(point_first[0] - point_second[0], 2)
				+ Math.pow(point_first[1] - point_second[1], 2));
	}

	public ListMap<double[], ThetaType> expectShortDistance(ListMap<double[], ThetaType> points_types, double[] center,
			double criterion_distance) {
		for (int i = 0; i < points_types.size(); i++) {
			// System.out.println("i: " + i);
			if (points_types.get(i).getValue().equals(ThetaType.FINGER)) {
				if (getBetweenDistance(points_types.get(i).getKey(), center) < criterion_distance * multiplie_amount) {
					points_types.remove(i);
				}
			}
		}
		return points_types;
	}

	public String getFilePath(String path) {
		return getFilePath(path, "");
	}

	public String getFilePath(String path, String add_text) {
		int last_index = 0;
		for (int i = 0; i < path.length(); i++) {
			if (path.charAt(i) == '.') last_index = i;
		}
		if (add_text.length() > 0) add_text = "_" + add_text;
		return path.substring(0, last_index) + add_text + "_" + System.currentTimeMillis()
				+ path.substring(last_index, path.length());
	}
}
