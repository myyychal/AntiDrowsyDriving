package com.mpanek.detection.nose;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.Objdetect;

import com.mpanek.constants.DrawingConstants;
import com.mpanek.detection.CascadeDetector;
import com.mpanek.utils.DrawingUtils;
import com.mpanek.utils.VisualUtils;

public class CascadeNoseDetector extends CascadeDetector {

	private Rect[] lastFoundNoses;
	private Rect lastFoundNose;

	public CascadeNoseDetector() {
		super();
		this.cascadeFileName = "haarcascade_mcs_nose.xml";
		TAG = "AntiDrowsyDriving::CascadeNoseDetector";
		this.scaleFactor = 1.2f;
		this.minNeighbours = 2;
		this.mRelativeMinObjectSize = 0.2f;
		this.mAbsoluteMinObjectSize = 0;
		this.mRelativeMaxObjectSize = 0.4f;
		this.mAbsoluteMaxObjectSize = 0;
		this.detectionFlag = Objdetect.CASCADE_DO_CANNY_PRUNING;
	}

	public Rect[] getLastFoundNoses() {
		return lastFoundNoses;
	}
	
	public Rect getLastFoundNose() {
		return lastFoundNose;
	}

	public Rect[] findNoses(Mat imgToFind, Rect face) {
		if (mAbsoluteMinObjectSize == 0 && mAbsoluteMaxObjectSize == 0) {
			int heightGray = face.height;
			if (Math.round(heightGray * mRelativeMinObjectSize) > 0 && Math.round(heightGray * mRelativeMaxObjectSize) > 0) {
				mAbsoluteMinObjectSize = Math.round(heightGray * mRelativeMinObjectSize);
				mAbsoluteMaxObjectSize = Math.round(heightGray * mRelativeMaxObjectSize);
			}
		}

		MatOfRect nose = new MatOfRect();

		Mat imgToFindWithROI;
		if (face != null) {
			face.height /= 2;
			face.y += face.height/2;
			imgToFindWithROI = new Mat(imgToFind, face);
		} else {
			imgToFindWithROI = imgToFind;
			face = new Rect();
		}
		
		if (javaDetector != null) {
			javaDetector.detectMultiScale(imgToFindWithROI, nose, scaleFactor, minNeighbours, detectionFlag, new Size(mAbsoluteMinObjectSize,
					mAbsoluteMinObjectSize), new Size(mAbsoluteMaxObjectSize, mAbsoluteMaxObjectSize));
		}

		Rect[] noseArray = nose.toArray();
		Rect[] newNoseArray = new Rect[noseArray.length];
		for (int i = 0; i < noseArray.length; i++) {
			Point noseTl = new Point(noseArray[i].x + face.x, noseArray[i].y + face.y);
			Point noseBr = new Point(noseArray[i].x + noseArray[i].width + face.x, noseArray[i].y + noseArray[i].height + face.y);
			newNoseArray[i] = new Rect(noseTl, noseBr);
		}

		if (newNoseArray != null && newNoseArray.length > 0) {
			lastFoundNoses = newNoseArray;
		}

		return newNoseArray;
	}
	
	public Rect findNose(Mat imgToFind, Rect face) {
		Rect[] nosesArray = findNoses(imgToFind, face);
		if (nosesArray != null) {
			Rect foundNose = null;
			double maxSurface = 0;
			for (Rect rect : nosesArray) {
				if (VisualUtils.calculateSurface(rect) > maxSurface) {
					maxSurface = VisualUtils.calculateSurface(rect);
					foundNose = rect;
				}
			}

			if (foundNose != null) {
				lastFoundNose = foundNose;
			}

			return foundNose;
		} else {
			return null;
		}
	}

}
