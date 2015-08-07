package com.mpanek.detection.elements.face;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.Objdetect;

import com.mpanek.detection.elements.CascadeDetector;
import com.mpanek.utils.VisualUtils;

public class CascadeFaceDetector extends CascadeDetector {

	private Rect lastFoundFace;

	public CascadeFaceDetector() {
		super();
		this.cascadeFileName = "lbpcascade_frontalface.xml";
		TAG = "AntiDrowsyDriving::CascadeFaceDetector";
		this.scaleFactor = 1.04f;
		this.minNeighbours = 2;
		this.mRelativeMinObjectSize = 0.35f;
		this.mAbsoluteMinObjectSize = 0;
		this.mRelativeMaxObjectSize = 0.75f;
		this.mAbsoluteMaxObjectSize = 0;
		this.detectionFlag = Objdetect.CASCADE_DO_CANNY_PRUNING;
	}

	public Rect getLastFoundFace() {
		return lastFoundFace;
	}

	public Rect findFace(Mat imgToFind, Rect rect) {
		Mat imgToFindWithROI;
		if (rect != null) {
			imgToFindWithROI = new Mat(imgToFind, rect);
		} else {
			imgToFindWithROI = imgToFind;
			rect = new Rect();
		}

		Rect foundFace = findFace(imgToFindWithROI);
		Rect foundFaceShifted = VisualUtils.shiftRectInRefToTOherRect(foundFace, rect);
		if (foundFaceShifted != null) {
			lastFoundFace = foundFaceShifted;
		}
		return foundFaceShifted;
	}

	public Rect findFace(Mat imgToFind) {
		if (mAbsoluteMinObjectSize == 0 && mAbsoluteMaxObjectSize == 0 || isSizeManuallyChanged) {
			int heightGray = imgToFind.rows();
			if (Math.round(heightGray * mRelativeMinObjectSize) > 0 && Math.round(heightGray * mRelativeMaxObjectSize) > 0) {
				mAbsoluteMinObjectSize = Math.round(heightGray * mRelativeMinObjectSize);
				mAbsoluteMaxObjectSize = Math.round(heightGray * mRelativeMaxObjectSize);
			}
			isSizeManuallyChanged = false;
		}

		MatOfRect faces = new MatOfRect();

		if (javaDetector != null) {
			javaDetector.detectMultiScale(imgToFind, faces, scaleFactor, minNeighbours, detectionFlag, new Size(mAbsoluteMinObjectSize,
					mAbsoluteMinObjectSize), new Size(mAbsoluteMaxObjectSize, mAbsoluteMaxObjectSize));
		}

		Rect mainFace = getMainFace(faces);
		if (mainFace != null) {
			lastFoundFace = mainFace;
		}

		return mainFace;
	}

	private Rect getMainFace(MatOfRect faces) {
		int maxFaceSize = 0;
		Rect chosenFace = null;
		for (Rect face : faces.toArray()) {
			if (face.width * face.height > maxFaceSize) {
				maxFaceSize = face.width * face.height;
				chosenFace = face;
			}
		}
		return chosenFace;
	}

}
