package com.mpanek.detection.main;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.mpanek.algorithms.ClaheAlgorithm;
import com.mpanek.constants.DrawingConstants;
import com.mpanek.detection.eyes.CascadeEyesDetector;
import com.mpanek.detection.face.CascadeFaceDetector;
import com.mpanek.detection.mouth.CascadeMouthDetector;
import com.mpanek.detection.nose.CascadeNoseDetector;
import com.mpanek.utils.DrawingUtils;

public class DrowsinessDetector {

	CascadeFaceDetector cascadeFaceDetector;
	CascadeEyesDetector cascadeEyesDetector;
	CascadeMouthDetector cascadeMouthDetector;
	CascadeNoseDetector cascadeNoseDetector;

	ClaheAlgorithm claheAlgorithm;

	final CharSequence[] items = { "Equalize histogram", "Gaussian blur",
			"Detect face", "Detect eyes", "Detect nose", "Detect mouth" };

	private boolean isEqualizeHistogram = true;
	private boolean isGaussianBlur = true;
	private boolean isDetectFace = true;
	private boolean isDetectEyes = true;
	private boolean isDetectNose = true;
	private boolean isDetectMouth = true;

	public DrowsinessDetector(CascadeFaceDetector cascadeFaceDetector,
			CascadeEyesDetector cascadeEyesDetector,
			CascadeMouthDetector cascadeMouthDetector,
			CascadeNoseDetector cascadeNoseDetector) {
		super();
		this.cascadeFaceDetector = cascadeFaceDetector;
		this.cascadeEyesDetector = cascadeEyesDetector;
		this.cascadeMouthDetector = cascadeMouthDetector;
		this.cascadeNoseDetector = cascadeNoseDetector;
		this.claheAlgorithm = new ClaheAlgorithm();
	}

	public DrowsinessDetector(CascadeFaceDetector cascadeFaceDetector,
			CascadeEyesDetector cascadeEyesDetector,
			CascadeMouthDetector cascadeMouthDetector,
			CascadeNoseDetector cascadeNoseDetector,
			ClaheAlgorithm claheAlgorithm) {
		this.cascadeFaceDetector = cascadeFaceDetector;
		this.cascadeEyesDetector = cascadeEyesDetector;
		this.cascadeMouthDetector = cascadeMouthDetector;
		this.cascadeNoseDetector = cascadeNoseDetector;
		this.claheAlgorithm = claheAlgorithm;
	}

	public Mat processDetection(Mat mGray, Mat mRgba) {
		if (isEqualizeHistogram) {
			claheAlgorithm.process(mGray);
		}
		if (isGaussianBlur) {
			Imgproc.GaussianBlur(mGray, mGray, new Size(5, 5), 0);
		}
		Rect foundFaceInDetection = new Rect(0, 0, mGray.width(),
				mGray.height());
		if (isDetectFace) {
			Rect boundingBox = new Rect(0, 0, mGray.width(), mGray.height());
			double boundingMultiplier = 0.1;
			boundingBox.x += boundingMultiplier * mGray.width();
			boundingBox.width -= 2 * boundingMultiplier * mGray.width();
			foundFaceInDetection = cascadeFaceDetector.findFace(mGray,
					boundingBox);
			if (foundFaceInDetection == null) {
				foundFaceInDetection = cascadeFaceDetector.getLastFoundFace();
			}
		}
		if (foundFaceInDetection != null) {
			Rect[] eyes = null;
			Rect[] mouths = null;
			Rect[] noses = null;
			if (isDetectEyes) {
				Rect foundFaceForEyes = foundFaceInDetection.clone();
				foundFaceForEyes.height /= 2;
				foundFaceForEyes.y = (int) (foundFaceForEyes.y + 0.1 * mGray
						.height());
				eyes = cascadeEyesDetector.findEyes(mGray, foundFaceForEyes);
				if (eyes == null || eyes.length == 0) {
					eyes = cascadeEyesDetector.getLastFoundEyes();
				}
			}
			if (isDetectMouth) {
				Rect foundFaceForMouth = foundFaceInDetection.clone();
				foundFaceForMouth.height /= 2;
				foundFaceForMouth.y = foundFaceForMouth.y
						+ foundFaceForMouth.height;
				mouths = cascadeMouthDetector.findMouths(mGray,
						foundFaceForMouth);
				if (mouths == null || mouths.length == 0) {
					mouths = cascadeMouthDetector.getLastFoundMouths();
				}
			}
			if (isDetectNose) {
				Rect foundFaceForNose = foundFaceInDetection.clone();
				foundFaceForNose.height /= 2;
				foundFaceForNose.y = foundFaceForNose.y
						+ foundFaceForNose.height;
				noses = cascadeNoseDetector.findNoses(mGray, foundFaceForNose);
				if (noses == null || noses.length == 0) {
					noses = cascadeNoseDetector.getLastFoundNoses();
				}
			}

			Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA);

			DrawingUtils.drawRect(foundFaceInDetection, mRgba,
					DrawingConstants.FACE_RECT_COLOR);

			DrawingUtils.drawRects(eyes, mRgba,
					DrawingConstants.EYES_RECT_COLOR);

			DrawingUtils.drawRects(mouths, mRgba,
					DrawingConstants.MOUTH_RECT_COLOR);

			DrawingUtils.drawRects(noses, mRgba,
					DrawingConstants.NOSE_RECT_COLOR);

		} else {
			Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA);
		}

		return mRgba;
	}

	public CascadeFaceDetector getCascadeFaceDetector() {
		return cascadeFaceDetector;
	}

	public void setCascadeFaceDetector(CascadeFaceDetector cascadeFaceDetector) {
		this.cascadeFaceDetector = cascadeFaceDetector;
	}

	public CascadeEyesDetector getCascadeEyesDetector() {
		return cascadeEyesDetector;
	}

	public void setCascadeEyesDetector(CascadeEyesDetector cascadeEyesDetector) {
		this.cascadeEyesDetector = cascadeEyesDetector;
	}

	public CascadeMouthDetector getCascadeMouthDetector() {
		return cascadeMouthDetector;
	}

	public void setCascadeMouthDetector(
			CascadeMouthDetector cascadeMouthDetector) {
		this.cascadeMouthDetector = cascadeMouthDetector;
	}

	public CascadeNoseDetector getCascadeNoseDetector() {
		return cascadeNoseDetector;
	}

	public void setCascadeNoseDetector(CascadeNoseDetector cascadeNoseDetector) {
		this.cascadeNoseDetector = cascadeNoseDetector;
	}

	public ClaheAlgorithm getClaheAlgorithm() {
		return claheAlgorithm;
	}

	public void setClaheAlgorithm(ClaheAlgorithm claheAlgorithm) {
		this.claheAlgorithm = claheAlgorithm;
	}

	public CharSequence[] getItems() {
		return items;
	}

	public void setDetectionElementsById(int id, boolean isChosen) {
		switch (id) {
		case 0:
			isEqualizeHistogram = isChosen;
			break;
		case 1:
			isGaussianBlur = isChosen;
			break;
		case 2:
			isDetectFace = isChosen;
			break;
		case 3:
			isDetectEyes = isChosen;
			break;
		case 4:
			isDetectNose = isChosen;
			break;
		case 5:
			isDetectNose = isChosen;
			break;
		}
	}

	public boolean isEqualizeHistogram() {
		return isEqualizeHistogram;
	}

	public void setEqualizeHistogram(boolean isEqualizeHistogram) {
		this.isEqualizeHistogram = isEqualizeHistogram;
	}

	public boolean isGaussianBlur() {
		return isGaussianBlur;
	}

	public void setGaussianBlur(boolean isGaussianBlur) {
		this.isGaussianBlur = isGaussianBlur;
	}

	public boolean isDetectFace() {
		return isDetectFace;
	}

	public void setDetectFace(boolean isDetectFace) {
		this.isDetectFace = isDetectFace;
	}

	public boolean isDetectEyes() {
		return isDetectEyes;
	}

	public void setDetectEyes(boolean isDetectEyes) {
		this.isDetectEyes = isDetectEyes;
	}

	public boolean isDetectNose() {
		return isDetectNose;
	}

	public void setDetectNose(boolean isDetectNose) {
		this.isDetectNose = isDetectNose;
	}

	public boolean isDetectMouth() {
		return isDetectMouth;
	}

	public void setDetectMouth(boolean isDetectMouth) {
		this.isDetectMouth = isDetectMouth;
	}

}
