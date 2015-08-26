package com.mpanek.algorithms.general;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import com.mpanek.constants.DrawingConstants;
import com.mpanek.utils.DrawingUtils;

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

	public void cannyEdgeDetection(Mat frame) {
		if (frame.channels() == 1) {
			Imgproc.Canny(frame, frame, firstThreshold, secondThreshold, apertureSize, isL2Gradient);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.Canny(frame, frame, firstThreshold, secondThreshold, apertureSize, isL2Gradient);
		}
	}

	public void sobelEdgeDetection(Mat frame) {
		if (frame.channels() == 1) {
			Imgproc.Sobel(frame, frame, CvType.CV_8U, 1, 1);
			Core.convertScaleAbs(frame, frame, 10, 0);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.Sobel(frame, frame, CvType.CV_8U, 1, 1);
			Core.convertScaleAbs(frame, frame, 10, 0);
		}
	}

	public void sobelAdvancedEdgeDetection(Mat frame) {
		if (frame.channels() == 1) {
			Imgproc.Sobel(frame, frame, CvType.CV_8U, 1, 1, 5, 1, 0);
			Core.convertScaleAbs(frame, frame, 10, 0);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.Sobel(frame, frame, CvType.CV_8U, 1, 1, 5, 1, 0);
			Core.convertScaleAbs(frame, frame, 10, 0);
		}
	}

	public void laplacianEdgeDetection(Mat frame) {
		if (frame.channels() == 1) {
			Imgproc.Laplacian(frame, frame, CvType.CV_8U);
			Core.convertScaleAbs(frame, frame, 13, 0);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.Laplacian(frame, frame, CvType.CV_8U);
			Core.convertScaleAbs(frame, frame, 13, 0);
		}
	}

	public void laplacianAdvancedEdgeDetection(Mat frame) {
		if (frame.channels() == 1) {
			Imgproc.Laplacian(frame, frame, CvType.CV_8U, 3, 1, 0);
			Core.convertScaleAbs(frame, frame, 10, 0);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.Laplacian(frame, frame, CvType.CV_8U, 3, 1, 0);
			Core.convertScaleAbs(frame, frame, 10, 0);
		}
	}

	public ArrayList<MatOfPoint> findContours(Mat frame) {
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		ArrayList<MatOfPoint> filteredContours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(frame, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
		for (MatOfPoint contour : contours) {
			if (Imgproc.boundingRect(contour).area() > 150) {
				filteredContours.add(contour);
			}
		}
		Imgproc.drawContours(frame, filteredContours, -1, DrawingConstants.PINK, 1);
		return filteredContours;
	}

	public Mat houghLines(Mat frame) {
		Mat lines = new Mat();
		Imgproc.HoughLinesP(frame, lines, 1, Math.PI / 180, 50, 0, 0);
		for (int i = 0; i < lines.cols(); i++) {
			double[] val = lines.get(0, i);
			Point[] twoPoints = new Point[] { new Point(val[0], val[1]), new Point(val[2], val[3]) };
			DrawingUtils.drawLines(twoPoints, frame, DrawingConstants.PINK);
		}
		return lines;
	}

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
