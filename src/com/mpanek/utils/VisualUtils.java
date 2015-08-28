package com.mpanek.utils;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import android.os.Environment;
import android.util.Log;

public class VisualUtils {

	private static final String TAG = "AntiDrowsyDriving::VisualUtils";
	public static boolean isPictureTakingAllowed = false;

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

	public static Rect shiftRectInRefToTOherRect(Rect originalRect, Rect referenceRect) {
		if (originalRect != null && referenceRect != null) {
			Point topLeftPoint = new Point(originalRect.x + referenceRect.x, originalRect.y + referenceRect.y);
			Point bottomRightPoint = new Point(originalRect.x + originalRect.width + referenceRect.x, originalRect.y + originalRect.height
					+ referenceRect.y);
			return new Rect(topLeftPoint, bottomRightPoint);
		} else {
			return null;
		}
	}

	public static double calculateSurface(Point tl, Point br) {
		return (br.x - tl.x) * (br.y - tl.y);
	}

	public static double calculateSurface(Rect rect) {
		Point tl = rect.tl();
		Point br = rect.br();
		return (br.x - tl.x) * (br.y - tl.y);
	}

	public static Point getCentrePoint(Rect rect) {
		return new Point((rect.tl().x + rect.br().x) / 2, (rect.tl().y + rect.br().y) / 2);
	}

	public static void resizeImage(Mat frame, float scale) {
		Imgproc.resize(frame, frame, new Size(scale * frame.width(), scale * frame.height()));
	}

	public static void takePicture(String fileName, Mat frameToSave, boolean isNameWithTimestamp) {
		if (isPictureTakingAllowed) {
			File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "AntiDrowsyDriving");

			if (!mediaStorageDir.exists()) {
				if (!mediaStorageDir.mkdirs()) {
					Log.e(TAG, "failed to create directory");
					return;
				}
			}

			File mediaFile;

			if (isNameWithTimestamp) {
				String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
				mediaFile = new File(mediaStorageDir.getPath() + File.separator + fileName + timeStamp + ".png");
			} else {
				mediaFile = new File(mediaStorageDir.getPath() + File.separator + fileName + ".png");
			}

			Highgui.imwrite(mediaFile.toString(), frameToSave);
		}
	}

}
