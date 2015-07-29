package com.mpanek.detection.nose;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.Objdetect;

import com.mpanek.detection.CascadeDetector;

public class CascadeNoseDetector extends CascadeDetector{
	
	private Rect[] lastFoundNoses;

	public CascadeNoseDetector() {
		super();
		this.cascadeFileName = "haarcascade_mcs_nose.xml";
		TAG = "AntiDrowsyDriving::CascadeNoseDetector";
		this.scaleFactor = 1.1f;
		this.minNeighbours = 2;
		this.mRelativeMinObjectSize = 0.1f;
		this.mAbsoluteMinObjectSize = 0;
		this.mRelativeMaxObjectSize = 0.5f;
		this.mAbsoluteMaxObjectSize = 0;
		this.detectionFlag = Objdetect.CASCADE_DO_CANNY_PRUNING;
	}

	public Rect[] getLastFoundNoses() {
		return lastFoundNoses;
	}

	public Rect[] findNoses(Mat imgToFind, Rect face) {
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

		MatOfRect nose = new MatOfRect();

		Mat imgToFindWithROI;
		if (face != null) {
			imgToFindWithROI = new Mat(imgToFind, face);
		} else {
			imgToFindWithROI = imgToFind;
			face = new Rect();
		}
		if (javaDetector != null) {
			javaDetector.detectMultiScale(imgToFindWithROI, nose, scaleFactor,
					minNeighbours,
					detectionFlag,
					new Size(mAbsoluteMinObjectSize, mAbsoluteMinObjectSize),
					new Size(mAbsoluteMaxObjectSize, mAbsoluteMaxObjectSize));
		}

		Rect[] noseArray = nose.toArray();
		Rect[] newNoseArray = new Rect[noseArray.length];
		for (int i = 0; i < noseArray.length; i++) {
			Point noseTl = new Point(noseArray[i].x + face.x, noseArray[i].y
					+ face.y);
			Point noseBr = new Point(noseArray[i].x + noseArray[i].width
					+ face.x, noseArray[i].y + noseArray[i].height + face.y);
			newNoseArray[i] = new Rect(noseTl, noseBr);
		}
		
		if (newNoseArray != null && newNoseArray.length > 0){
			lastFoundNoses = newNoseArray;
		}
		
		return newNoseArray;
	}

}
