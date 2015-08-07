package com.mpanek.detection.main;

import java.util.ArrayList;

import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.mpanek.algorithms.general.ClaheAlgorithm;
import com.mpanek.constants.DrawingConstants;
import com.mpanek.detection.elements.eyes.CascadeEyesDetector;
import com.mpanek.detection.elements.face.CascadeFaceDetector;
import com.mpanek.detection.elements.mouth.CascadeMouthDetector;
import com.mpanek.detection.elements.nose.CascadeNoseDetector;
import com.mpanek.utils.DrawingUtils;

public class DrowsinessDetector {

	CascadeFaceDetector cascadeFaceDetector;
	CascadeEyesDetector cascadeEyesDetector;
	CascadeMouthDetector cascadeMouthDetector;
	CascadeNoseDetector cascadeNoseDetector;

	CascadeEyesDetector cascadeLeftEyeDetector;
	CascadeEyesDetector cascadeRightEyeDetector;

	ClaheAlgorithm claheAlgorithm;

	final CharSequence[] items = { "Equalize histogram", "Gaussian blur", "Detect face", "Detect eyes", "Detect nose", "Detect mouth",
			"Additional equalization after face detection", "Additional gaussian blur after face detection" };

	private boolean isEqualizeHistogram = true;
	private boolean isGaussianBlur = true;
	private boolean isDetectFace = true;
	private boolean isAdditionalEqualization = false;
	private boolean isAdditionalGauss = false;
	private boolean isDetectEyes = true;
	private boolean isDetectNose = true;
	private boolean isDetectMouth = true;

	private boolean isSeparateEyesDetection = false;

	int gaussianBlur = 5;

	public DrowsinessDetector(CascadeFaceDetector cascadeFaceDetector, CascadeEyesDetector cascadeEyesDetector,
			CascadeMouthDetector cascadeMouthDetector, CascadeNoseDetector cascadeNoseDetector) {
		super();
		this.cascadeFaceDetector = cascadeFaceDetector;
		this.cascadeEyesDetector = cascadeEyesDetector;
		this.cascadeMouthDetector = cascadeMouthDetector;
		this.cascadeNoseDetector = cascadeNoseDetector;
		this.claheAlgorithm = new ClaheAlgorithm();
	}

	public DrowsinessDetector(CascadeFaceDetector cascadeFaceDetector, CascadeEyesDetector cascadeEyesDetector,
			CascadeMouthDetector cascadeMouthDetector, CascadeNoseDetector cascadeNoseDetector, ClaheAlgorithm claheAlgorithm) {
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
			Imgproc.GaussianBlur(mGray, mGray, new Size(gaussianBlur, gaussianBlur), 0);
		}
		Rect foundFaceInDetection = new Rect(0, 0, mGray.width(), mGray.height());
		if (isDetectFace) {
			Rect boundingBox = new Rect(0, 0, mGray.width(), mGray.height());
			double boundingMultiplier = 0.1;
			boundingBox.x += boundingMultiplier * mGray.width();
			boundingBox.width -= 2 * boundingMultiplier * mGray.width();
			foundFaceInDetection = cascadeFaceDetector.findFace(mGray, boundingBox);
			if (foundFaceInDetection == null) {
				foundFaceInDetection = cascadeFaceDetector.getLastFoundFace();
			}
		}
		if (foundFaceInDetection != null) {
			Rect[] eyes = null;
			Rect mouth = null;
			Rect nose = null;

			Rect[] leftEyes = null;
			Rect[] rightEyes = null;

			Mat imgToFindWithROI;
			try {
				imgToFindWithROI = new Mat(mGray, foundFaceInDetection);
			} catch (CvException e) {
				return mRgba;
			}

			if (isAdditionalEqualization) {
				ClaheAlgorithm claheAlgorithm = new ClaheAlgorithm();
				claheAlgorithm.process(imgToFindWithROI);
			}
			if (isAdditionalGauss) {
				Imgproc.GaussianBlur(mGray, mGray, new Size(gaussianBlur, gaussianBlur), 0);
			}

			if (isDetectEyes) {
				if (isSeparateEyesDetection) {
					Rect foundFaceForLeftEyes = foundFaceInDetection.clone();
					foundFaceForLeftEyes.width /= 2;
					leftEyes = cascadeLeftEyeDetector.findEyes(mGray, foundFaceForLeftEyes);
					if (leftEyes == null || leftEyes.length == 0) {
						leftEyes = cascadeEyesDetector.getLastFoundEyes();
					}
					Rect foundFaceForRightEyes = foundFaceInDetection.clone();
					foundFaceForRightEyes.width /= 2;
					foundFaceForRightEyes.x += foundFaceForRightEyes.width;
					rightEyes = cascadeRightEyeDetector.findEyes(mGray, foundFaceForRightEyes);
					if (rightEyes == null || rightEyes.length == 0) {
						rightEyes = cascadeEyesDetector.getLastFoundEyes();
					}
				} else {
					Rect foundFaceForEyes = foundFaceInDetection.clone();
					eyes = cascadeEyesDetector.findEyes(mGray, foundFaceForEyes);
					if (eyes == null || eyes.length == 0) {
						eyes = cascadeEyesDetector.getLastFoundEyes();
					}
				}
			}
			if (isDetectMouth) {
				Rect foundFaceForMouth = foundFaceInDetection.clone();
				mouth = cascadeMouthDetector.findMouth(mGray, foundFaceForMouth);
				if (mouth == null) {
					mouth = cascadeMouthDetector.getLastFoundMouth();
				}
			}
			if (isDetectNose) {
				Rect foundFaceForNose = foundFaceInDetection.clone();
				nose = cascadeNoseDetector.findNose(mGray, foundFaceForNose);
				if (nose == null) {
					nose = cascadeNoseDetector.getLastFoundNose();
				}
			}

			Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA);

			DrawingUtils.drawRect(foundFaceInDetection, mRgba, DrawingConstants.FACE_RECT_COLOR);

			if (isSeparateEyesDetection) {
				DrawingUtils.drawRects(leftEyes, mRgba, DrawingConstants.EYES_RECT_COLOR);
				DrawingUtils.drawRects(rightEyes, mRgba, DrawingConstants.EYES_RECT_COLOR);
			} else {
				DrawingUtils.drawRects(eyes, mRgba, DrawingConstants.EYES_RECT_COLOR);
			}

			DrawingUtils.drawRect(mouth, mRgba, DrawingConstants.MOUTH_RECT_COLOR);

			DrawingUtils.drawRect(nose, mRgba, DrawingConstants.NOSE_RECT_COLOR);

			ArrayList<Rect> allDetectedRects = new ArrayList<Rect>();
			if (foundFaceInDetection != null) {
				allDetectedRects.add(foundFaceInDetection);
			}
			if (isSeparateEyesDetection) {
				if (leftEyes != null) {
					for (Rect eyeRect : leftEyes) {
						if (eyeRect != null) {
							allDetectedRects.add(eyeRect);
						}
					}
				}
				if (rightEyes != null) {
					for (Rect eyeRect : rightEyes) {
						if (eyeRect != null) {
							allDetectedRects.add(eyeRect);
						}
					}
				}
			} else {
				if (eyes != null) {
					for (Rect eyeRect : eyes) {
						if (eyeRect != null) {
							allDetectedRects.add(eyeRect);
						}
					}
				}
			}
			if (mouth != null) {
				allDetectedRects.add(mouth);
			}
			if (nose != null) {
				allDetectedRects.add(nose);
			}
			DrawingUtils.drawLinesFromRectanglesCentres(allDetectedRects.toArray(new Rect[allDetectedRects.size()]), mRgba, DrawingConstants.WHITE);

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

	public void setCascadeMouthDetector(CascadeMouthDetector cascadeMouthDetector) {
		this.cascadeMouthDetector = cascadeMouthDetector;
	}

	public CascadeNoseDetector getCascadeNoseDetector() {
		return cascadeNoseDetector;
	}

	public void setCascadeNoseDetector(CascadeNoseDetector cascadeNoseDetector) {
		this.cascadeNoseDetector = cascadeNoseDetector;
	}

	public CascadeEyesDetector getCascadeLeftEyeDetector() {
		return cascadeLeftEyeDetector;
	}

	public void setCascadeLeftEyeDetector(CascadeEyesDetector cascadeLeftEyeDetector) {
		this.cascadeLeftEyeDetector = cascadeLeftEyeDetector;
	}

	public CascadeEyesDetector getCascadeRightEyeDetector() {
		return cascadeRightEyeDetector;
	}

	public void setCascadeRightEyeDetector(CascadeEyesDetector cascadeRightEyeDetector) {
		this.cascadeRightEyeDetector = cascadeRightEyeDetector;
	}

	public boolean isSeparateEyesDetection() {
		return isSeparateEyesDetection;
	}

	public void setSeparateEyesDetection(boolean isSeparateEyesDetection) {
		this.isSeparateEyesDetection = isSeparateEyesDetection;
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
			isDetectMouth = isChosen;
			break;
		case 6:
			isAdditionalEqualization = isChosen;
			break;
		case 7:
			isAdditionalGauss = isChosen;
			break;
		}
	}

	public void setAllDetectionElements(boolean value) {
		isEqualizeHistogram = value;
		isGaussianBlur = value;
		isDetectFace = value;
		isDetectEyes = value;
		isDetectNose = value;
		isDetectMouth = value;
	}

	public boolean[] getDetectionFlags() {
		boolean[] checks = new boolean[8];
		checks[0] = isEqualizeHistogram;
		checks[1] = isGaussianBlur;
		checks[2] = isDetectFace;
		checks[3] = isDetectEyes;
		checks[4] = isDetectNose;
		checks[5] = isDetectMouth;
		checks[6] = isAdditionalEqualization;
		checks[7] = isAdditionalEqualization;
		return checks;

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

	public boolean isAdditionalEqualization() {
		return isAdditionalEqualization;
	}

	public void setAdditionalEqualization(boolean isAdditionalEqualization) {
		this.isAdditionalEqualization = isAdditionalEqualization;
	}

	public boolean isAdditionalGauss() {
		return isAdditionalGauss;
	}

	public void setAdditionalGauss(boolean isAdditionalGauss) {
		this.isAdditionalGauss = isAdditionalGauss;
	}

	public int getGaussianBlur() {
		return gaussianBlur;
	}

	public void setGaussianBlur(int gaussianBlur) {
		this.gaussianBlur = gaussianBlur;
	}

}
