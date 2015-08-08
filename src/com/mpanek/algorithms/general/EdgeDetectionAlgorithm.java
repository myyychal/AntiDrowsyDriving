package com.mpanek.algorithms.general;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class EdgeDetectionAlgorithm {
	
	private static final String TAG = "AntiDrowsyDriving::EdgeDetectionAlgorithm";
	
	int firstThreshold;
	int secondThreshold;
	int apertureSize;
	boolean isL2Gradient;
	
	public EdgeDetectionAlgorithm() {
		this.firstThreshold = 80;
		this.secondThreshold = 160;
		this.apertureSize = 3;
		this.isL2Gradient = false;
	}

	public EdgeDetectionAlgorithm(int firstThreshold, int secondThreshold, int apertureSize, boolean isL2Gradient) {
		this.firstThreshold = firstThreshold;
		this.secondThreshold = secondThreshold;
		this.apertureSize = apertureSize;
		this.isL2Gradient = isL2Gradient;
	}

	public void cannyEdgeDetection(Mat frame){
		if (frame.channels() == 1){
			Imgproc.Canny(frame, frame, firstThreshold, secondThreshold, apertureSize, isL2Gradient);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.Canny(frame, frame, firstThreshold, secondThreshold, apertureSize, isL2Gradient);
		}

	}
	
	//findContours
	//houghLines

	public int getFirstThreshold() {
		return firstThreshold;
	}

	public void setFirstThreshold(int firstThreshold) {
		this.firstThreshold = firstThreshold;
	}

	public int getSecondThreshold() {
		return secondThreshold;
	}

	public void setSecondThreshold(int secondThreshold) {
		this.secondThreshold = secondThreshold;
	}

	public int getApertureSize() {
		return apertureSize;
	}

	public void setApertureSize(int apertureSize) {
		this.apertureSize = apertureSize;
	}

	public boolean isL2Gradient() {
		return isL2Gradient;
	}

	public void setL2Gradient(boolean isL2Gradient) {
		this.isL2Gradient = isL2Gradient;
	}
	
}
