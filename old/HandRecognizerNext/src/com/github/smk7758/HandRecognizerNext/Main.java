package com.github.smk7758.HandRecognizerNext;

import java.nio.file.Files;
import java.nio.file.Paths;

import org.opencv.core.Core;

public class Main {
	private final String video_path = "S:\\NextHR\\Result_2018-11-13\\CIMG1085.MOV"; // TODO
	// private long start_time = System.currentTimeMillis();

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("START");
	}

	public static void main(String[] args) {
		new Main().lunchProcesser();

	}

	public void lunchProcesser() {
		if (Files.exists(Paths.get(video_path))) {
			try (Processer processer = new Processer(video_path)) {
				processer.recognizer();
			}
		} else {
			System.err.println("???");
		}
	}
}