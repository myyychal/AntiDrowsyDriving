package com.mpanek.detection.face;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class ColorSegmentationFaceDetector {

	private String TAG = "ColorSegmentationFaceDetector";

	public Mat detectFaceYCrCb(Mat rgbFrame) {

		Mat yCrBrFrame = new Mat(rgbFrame.height(), rgbFrame.width(), CvType.CV_8UC4);
		Imgproc.cvtColor(rgbFrame, yCrBrFrame, Imgproc.COLOR_RGB2YCrCb);

		Mat inRangeResultFrame = new Mat();
		Core.inRange(yCrBrFrame, new Scalar(0, 133, 77), new Scalar(255, 173, 127), inRangeResultFrame);

		return inRangeResultFrame;

	}

	public Mat detectFaceHSV(Mat rgbFrame) {

		Mat yCrBrFrame = new Mat(rgbFrame.height(), rgbFrame.width(), CvType.CV_8UC4);
		Imgproc.cvtColor(rgbFrame, yCrBrFrame, Imgproc.COLOR_RGB2HSV_FULL);

		Mat inRangeResultFrame = new Mat();
		Core.inRange(yCrBrFrame, new Scalar(0, 200, 0), new Scalar(25, 216, 255), inRangeResultFrame);

		return inRangeResultFrame;

	}

}
