package com.mpanek.detection.elements.mouth;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.Objdetect;

import com.mpanek.detection.elements.CascadeDetector;

public class CascadeMouthDetector extends CascadeDetector {
	
	private static final String TAG = "AntiDrowsyDriving::CascadeMouthDetector";

	private Rect[] lastFoundMouths;
	private Rect lastFoundMouth;

	public CascadeMouthDetector() {
		super();
		this.cascadeFileName = "haarcascade_mcs_mouth.xml";
		this.scaleFactor = 1.2f;
		this.minNeighbours = 2;
		this.mRelativeMinObjectSize = 0.1f;
		this.mAbsoluteMinObjectSize = 0;
		this.mRelativeMaxObjectSize = 0.4f;
		this.mAbsoluteMaxObjectSize = 0;
		this.detectionFlag = Objdetect.CASCADE_DO_CANNY_PRUNING;
	}

	public Rect[] getLastFoundMouths() {
		return lastFoundMouths;
	}

	public Rect getLastFoundMouth() {
		return lastFoundMouth;
	}

	public Rect[] findMouths(Mat imgToFind, Rect face) {
		if (mAbsoluteMinObjectSize == 0 && mAbsoluteMaxObjectSize == 0) {
			int heightGray = face.height;
			if (Math.round(heightGray * mRelativeMinObjectSize) > 0 && Math.round(heightGray * mRelativeMaxObjectSize) > 0) {
				mAbsoluteMinObjectSize = Math.round(heightGray * mRelativeMinObjectSize);
				mAbsoluteMaxObjectSize = Math.round(heightGray * mRelativeMaxObjectSize);
			}
		}

		MatOfRect mouth = new MatOfRect();

		Mat imgToFindWithROI;
		if (face != null) {
			face.height /= 2;
			face.y = face.y + face.height;
			imgToFindWithROI = new Mat(imgToFind, face);
		} else {
			imgToFindWithROI = imgToFind;
			face = new Rect();
		}
		if (javaDetector != null) {
			javaDetector.detectMultiScale(imgToFindWithROI, mouth, scaleFactor, minNeighbours, detectionFlag, new Size(mAbsoluteMinObjectSize,
					mAbsoluteMinObjectSize), new Size(mAbsoluteMaxObjectSize, mAbsoluteMaxObjectSize));
		}

		Rect[] mouthArray = mouth.toArray();
		Rect[] newMouthArray = new Rect[mouthArray.length];
		for (int i = 0; i < mouthArray.length; i++) {
			Point mouthTl = new Point(mouthArray[i].x + face.x, mouthArray[i].y + face.y);
			Point mouthBr = new Point(mouthArray[i].x + mouthArray[i].width + face.x, mouthArray[i].y + mouthArray[i].height + face.y);
			newMouthArray[i] = new Rect(mouthTl, mouthBr);
		}

		if (newMouthArray != null && newMouthArray.length > 0) {
			lastFoundMouths = newMouthArray;
		}

		return newMouthArray;
	}

	public Rect findMouth(Mat imgToFind, Rect face) {
		Rect[] mouthArray = findMouths(imgToFind, face);
		if (mouthArray != null) {
			Rect foundMouth = null;
			double maxY = 0;
			for (Rect rect : mouthArray) {
				if (rect.br().y > maxY) {
					maxY = rect.br().y;
					foundMouth = rect;
				}
			}

			if (foundMouth != null) {
				lastFoundMouth = foundMouth;
			}

			return foundMouth;
		} else {
			return null;
		}
	}

}
