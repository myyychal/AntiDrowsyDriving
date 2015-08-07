package com.mpanek.algorithms.general;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class HistogramEqualizationAlgorithm {

	public void standardEqualizationJava(Mat frame){
		if (frame.channels() == 1){
			Imgproc.equalizeHist(frame, frame);
		} else {
			Mat yCrCb = new Mat(frame.size(), CvType.CV_8UC3);
			Imgproc.cvtColor(frame, yCrCb, Imgproc.COLOR_RGB2YCrCb);
			ArrayList<Mat> channels = new ArrayList<Mat>();
			Core.split(yCrCb, channels);
			Imgproc.equalizeHist(channels.get(0), channels.get(0));
			Core.merge(channels, yCrCb);
			Imgproc.cvtColor(yCrCb, frame, Imgproc.COLOR_YCrCb2RGB);
		}
	}
	
	public void standardEqualizationCpp(Mat frame){
		EqualizeHistogram(frame.getNativeObjAddr());
	}
	
	public native void EqualizeHistogram(long matAddrGr);
	
}
