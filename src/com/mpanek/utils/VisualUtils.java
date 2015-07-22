package com.mpanek.utils;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class VisualUtils {

	public static Mat equalizeIntensity(Mat inputImage) {
		if (inputImage.channels() >= 3) {
			Mat ycrcb = new Mat();

			Imgproc.cvtColor(inputImage, ycrcb, Imgproc.COLOR_RGB2YCrCb);

			ArrayList<Mat> channels = new ArrayList<Mat>();
			Core.split(ycrcb, channels);

			Imgproc.equalizeHist(channels.get(0), channels.get(0));

			Mat result = new Mat();
			Core.merge(channels, ycrcb);

			Imgproc.cvtColor(ycrcb, result, Imgproc.COLOR_YCrCb2RGB);

			return result;
		} else {
			return inputImage;
		}
	}

	public static Rect shiftRectInRefToTOherRect(Rect originalRect,
			Rect referenceRect) {
		if (originalRect != null && referenceRect != null) {
			Point topLeftPoint = new Point(originalRect.x + referenceRect.x,
					originalRect.y + referenceRect.y);
			Point bottomRightPoint = new Point(originalRect.x
					+ originalRect.width + referenceRect.x, originalRect.y
					+ originalRect.height + referenceRect.y);
			return new Rect(topLeftPoint, bottomRightPoint);
		} else {
			return null;
		}
	}

}
