package com.mpanek.algorithms.general;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class BinarizationAlgorithm {
	
	private static final String TAG = "AntiDrowsyDriving::BinarizationAlgorithm";
	
	int threshold;
	int maxValue;
	int blockSize;
	double C;
	
	public BinarizationAlgorithm(){
		this.threshold = 127;
		this.maxValue = 255;
		this.blockSize = 15;
		C = 4;
	}
	
	public BinarizationAlgorithm(int threshold, int maxValue, int blockSize, double c) {
		this.threshold = threshold;
		this.maxValue = maxValue;
		this.blockSize = blockSize;
		C = c;
	}

	public void standardBinarization(Mat frame){
		if (frame.channels() == 1){
			Imgproc.threshold(frame, frame, threshold , maxValue, Imgproc.THRESH_BINARY);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.threshold(frame, frame, threshold , maxValue, Imgproc.THRESH_BINARY);
		}
	}
	
	public void standardTruncBinarization(Mat frame){
		if (frame.channels() == 1){
			Imgproc.threshold(frame, frame, threshold , maxValue, Imgproc.THRESH_TRUNC);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.threshold(frame, frame, threshold , maxValue, Imgproc.THRESH_TRUNC);
		}
	}
	
	public void otsuBinarization(Mat frame){
		if (frame.channels() == 1){
			Imgproc.threshold(frame, frame, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.threshold(frame, frame, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
		}
	}
	
	public void adaptiveMeanBinarization(Mat frame){
		if (frame.channels() == 1){
			Imgproc.adaptiveThreshold(frame, frame, 255,
			         Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, blockSize, C);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.adaptiveThreshold(frame, frame, 255,
			         Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, blockSize, C);
		}
	}
	
	public void adaptiveGaussianBinarization(Mat frame){
		if (frame.channels() == 1){
			Imgproc.adaptiveThreshold(frame, frame, 255,
			         Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, blockSize, C);
		} else {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
			Imgproc.adaptiveThreshold(frame, frame, 255,
			         Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, blockSize, C);
		}
	}

	public int getThreshold() {
		return threshold;
	}

	public void setThreshold(int threshold) {
		this.threshold = threshold;
	}

	public int getMaxValue() {
		return maxValue;
	}

	public void setMaxValue(int maxValue) {
		this.maxValue = maxValue;
	}

	public int getBlockSize() {
		return blockSize;
	}

	public void setBlockSize(int blockSize) {
		this.blockSize = blockSize;
	}

	public double getC() {
		return C;
	}

	public void setC(double c) {
		C = c;
	}
	
}
