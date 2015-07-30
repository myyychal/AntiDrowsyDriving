package com.mpanek.detection.eyes;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.Objdetect;

import android.util.Log;

import com.mpanek.detection.CascadeDetector;

public class CascadeEyesDetector extends CascadeDetector{

	private Rect[] lastFoundEyes;
	
	public CascadeEyesDetector() {
		super();
		cascadeFileName = "haarcascade_eye_tree_eyeglasses.xml";
		TAG = "AntiDrowsyDriving::CascadeEyesDetector";
		this.scaleFactor = 1.1f;
		this.minNeighbours = 2;
		this.mRelativeMinObjectSize = 0.15f;
		this.mAbsoluteMinObjectSize = 0;
		this.mRelativeMaxObjectSize = 0.3f;
		this.mAbsoluteMaxObjectSize = 0;
		this.detectionFlag = Objdetect.CASCADE_DO_CANNY_PRUNING;
	}
	
	public Rect[] getLastFoundEyes() {
		if (lastFoundEyes != null){
			Log.i(TAG, "Last found eyes size: " + String.valueOf(lastFoundEyes.length));
		}
		return lastFoundEyes;
	}

	public Rect[] findEyes(Mat imgToFind) {
		if (mAbsoluteMinObjectSize == 0 && mAbsoluteMaxObjectSize == 0) {
			int heightGray = imgToFind.rows();
			if (Math.round(heightGray * mRelativeMinObjectSize) > 0
					&& Math.round(heightGray * mRelativeMaxObjectSize) > 0) {
				mAbsoluteMinObjectSize = Math.round(heightGray
						* mRelativeMinObjectSize);
				mAbsoluteMaxObjectSize = Math.round(heightGray
						* mRelativeMaxObjectSize);
			}
		}

		MatOfRect eyes = new MatOfRect();

		if (javaDetector != null) {
			javaDetector.detectMultiScale(imgToFind, eyes, scaleFactor,
					minNeighbours,
					detectionFlag,
					new Size(mAbsoluteMinObjectSize, mAbsoluteMinObjectSize),
					new Size(mAbsoluteMaxObjectSize, mAbsoluteMaxObjectSize));
		}

		Rect[] eyesArray = eyes.toArray();
		
		if (eyesArray != null && eyesArray.length > 0){
			lastFoundEyes = eyesArray;
		}

		return eyesArray;
	}

	public Rect[] findEyes(Mat imgToFind, Rect face) {
		if (mAbsoluteMinObjectSize == 0 && mAbsoluteMaxObjectSize == 0
				|| isSizeManuallyChanged) {
			int heightGray = face.height;
			if (Math.round(heightGray * mRelativeMinObjectSize) > 0
					&& Math.round(heightGray * mRelativeMaxObjectSize) > 0) {
				mAbsoluteMinObjectSize = Math.round(heightGray
						* mRelativeMinObjectSize);
				mAbsoluteMaxObjectSize = Math.round(heightGray
						* mRelativeMaxObjectSize);
			}
			isSizeManuallyChanged = false;
		}

		MatOfRect eyes = new MatOfRect();

		Mat imgToFindWithROI;
		if (face != null) {
			face.height /= 2;
			face.y = (int) (face.y + 0.1 * imgToFind
					.height());
			imgToFindWithROI = new Mat(imgToFind, face);
		} else {
			imgToFindWithROI = imgToFind;
			face = new Rect();
		}
		
		if (javaDetector != null) {
			javaDetector.detectMultiScale(imgToFindWithROI, eyes, scaleFactor,
					minNeighbours,
					detectionFlag,
					new Size(mAbsoluteMinObjectSize, mAbsoluteMinObjectSize),
					new Size(mAbsoluteMaxObjectSize, mAbsoluteMaxObjectSize));
		}

		Rect[] eyesArray = eyes.toArray();
		Rect[] newEyesArray = new Rect[eyesArray.length];
		for (int i = 0; i < eyesArray.length; i++) {
			Point eyeTl = new Point(eyesArray[i].x + face.x, eyesArray[i].y
					+ face.y);
			Point eyeBr = new Point(eyesArray[i].x + eyesArray[i].width
					+ face.x, eyesArray[i].y + eyesArray[i].height + face.y);
			newEyesArray[i] = new Rect(eyeTl, eyeBr);
		}
		
		if (newEyesArray != null && newEyesArray.length > 0){
			lastFoundEyes = newEyesArray;
		}
		
		return newEyesArray;
	}

}
