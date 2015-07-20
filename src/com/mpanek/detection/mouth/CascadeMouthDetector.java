package com.mpanek.detection.mouth;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.Objdetect;

import com.mpanek.detection.CascadeDetector;

public class CascadeMouthDetector extends CascadeDetector{

	public CascadeMouthDetector() {
		super();
		this.cascadeFileName = "haarcascade_mcs_mouth.xml";
		TAG = "AntiDrowsyDriving::CascadeMouthDetector";
		this.scaleFactor = 1.1f;
		this.minNeighbours = 2;
		this.mRelativeMinObjectSize = 0.1f;
		this.mAbsoluteMinObjectSize = 0;
		this.mRelativeMaxObjectSize = 0.5f;
		this.mAbsoluteMaxObjectSize = 0;
		this.detectionFlag = Objdetect.CASCADE_DO_CANNY_PRUNING;
	}

	public Rect[] findMouth(Mat imgToFind, Rect face) {
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

		MatOfRect mouth = new MatOfRect();

		Mat imgToFindWithROI;
		if (face != null) {
			imgToFindWithROI = new Mat(imgToFind, face);
		} else {
			imgToFindWithROI = imgToFind;
			face = new Rect();
		}
		if (javaDetector != null) {
			javaDetector.detectMultiScale(imgToFindWithROI, mouth, scaleFactor,
					minNeighbours,
					detectionFlag,
					new Size(mAbsoluteMinObjectSize, mAbsoluteMinObjectSize),
					new Size(mAbsoluteMaxObjectSize, mAbsoluteMaxObjectSize));
		}

		Rect[] mouthArray = mouth.toArray();
		Rect[] newMouthArray = new Rect[mouthArray.length];
		for (int i = 0; i < mouthArray.length; i++) {
			Point mouthTl = new Point(mouthArray[i].x + face.x, mouthArray[i].y
					+ face.y);
			Point mouthBr = new Point(mouthArray[i].x + mouthArray[i].width
					+ face.x, mouthArray[i].y + mouthArray[i].height + face.y);
			newMouthArray[i] = new Rect(mouthTl, mouthBr);
		}
		
		return newMouthArray;
	}

}
