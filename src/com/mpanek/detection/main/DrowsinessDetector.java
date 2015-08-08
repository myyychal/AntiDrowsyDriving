package com.mpanek.detection.main;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.util.Log;

import com.mpanek.algorithms.general.ClaheAlgorithm;
import com.mpanek.algorithms.general.EdgeDetectionAlgorithm;
import com.mpanek.algorithms.specialized.DarkBrightRatioAlgorithm;
import com.mpanek.constants.DrawingConstants;
import com.mpanek.detection.elements.eyes.CascadeEyesDetector;
import com.mpanek.detection.elements.face.CascadeFaceDetector;
import com.mpanek.detection.elements.mouth.CascadeMouthDetector;
import com.mpanek.detection.elements.nose.CascadeNoseDetector;
import com.mpanek.utils.DrawingUtils;
import com.mpanek.utils.VisualUtils;

public class DrowsinessDetector {
	
	private static final String TAG = "AntiDrowsyDriving::DrowsinessDetector";

	CascadeFaceDetector cascadeFaceDetector;
	CascadeEyesDetector cascadeEyesDetector;
	CascadeMouthDetector cascadeMouthDetector;
	CascadeNoseDetector cascadeNoseDetector;

	CascadeEyesDetector cascadeLeftEyeDetector;
	CascadeEyesDetector cascadeRightEyeDetector;

	ClaheAlgorithm claheAlgorithm;
	DarkBrightRatioAlgorithm darkBrightRatioAlgorithm;
	EdgeDetectionAlgorithm edgeDetectionAlgorithm;

	final CharSequence[] items = { "Equalize histogram", "Gaussian blur", "Detect face", "Detect eyes", "Detect nose", "Detect mouth",
			"Additional equalization after face detection", "Additional gaussian blur after face detection" };

	private boolean isEqualizeHistogram = true;
	private boolean isGaussianBlur = true;
	private boolean isDetectFace = true;
	private boolean isAdditionalEqualization = true;
	private boolean isAdditionalGauss = false;
	private boolean isDetectEyes = true;
	private boolean isDetectNose = false;
	private boolean isDetectMouth = false;

	private boolean isSeparateEyesDetection = false;
	private boolean isCannyAlgorithmUsed = false;
	private boolean isSobelAlgorithmUsed = false;
	private boolean isSimpleBinarizationUsed = false;

	private boolean isDoNothing = false;

	int gaussianBlur = 5;

	long frameCounter = 0;
	boolean isFaceFound = false;

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
			CascadeMouthDetector cascadeMouthDetector, CascadeNoseDetector cascadeNoseDetector, ClaheAlgorithm claheAlgorithm,
			DarkBrightRatioAlgorithm darkBrightRatioAlgorithm, EdgeDetectionAlgorithm edgeDetectionAlgorithm) {
		this.cascadeFaceDetector = cascadeFaceDetector;
		this.cascadeEyesDetector = cascadeEyesDetector;
		this.cascadeMouthDetector = cascadeMouthDetector;
		this.cascadeNoseDetector = cascadeNoseDetector;
		this.claheAlgorithm = claheAlgorithm;
		this.darkBrightRatioAlgorithm = darkBrightRatioAlgorithm;
		// darkBrightRatioAlgorithm.getBinarizationAlgorithm().setBlockSize(3);
		// darkBrightRatioAlgorithm.getBinarizationAlgorithm().setC(1.12);
		// darkBrightRatioAlgorithm.setErosionSize(3);
		darkBrightRatioAlgorithm.getBinarizationAlgorithm().setBlockSize(35);
		this.edgeDetectionAlgorithm = edgeDetectionAlgorithm;
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
			if (frameCounter < 5 || frameCounter % 10 == 0 || !isFaceFound) {
				foundFaceInDetection = cascadeFaceDetector.findFace(mGray, boundingBox);
				if (foundFaceInDetection == null) {
					isFaceFound = false;
				} else {
					isFaceFound = true;
				}
			} else {
				foundFaceInDetection = null;
			}
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
				claheAlgorithm.process(imgToFindWithROI);
			}
			if (isAdditionalGauss) {
				Imgproc.GaussianBlur(mGray, mGray, new Size(gaussianBlur, gaussianBlur), 0);
			}

			if (isDetectEyes) {
				if (isSeparateEyesDetection) {
					// swapped detectors
					Rect foundFaceForLeftEyes = foundFaceInDetection.clone();
					foundFaceForLeftEyes.width /= 2;
					leftEyes = cascadeRightEyeDetector.findEyes(mGray, foundFaceForLeftEyes, true);
					if (leftEyes == null || leftEyes.length == 0) {
						leftEyes = cascadeRightEyeDetector.getLastFoundEyes();
					}
					Rect foundFaceForRightEyes = foundFaceInDetection.clone();
					foundFaceForRightEyes.width /= 2;
					foundFaceForRightEyes.x += foundFaceForRightEyes.width;
					rightEyes = cascadeLeftEyeDetector.findEyes(mGray, foundFaceForRightEyes, true);
					if (rightEyes == null || rightEyes.length == 0) {
						rightEyes = cascadeLeftEyeDetector.getLastFoundEyes();
					}
					ArrayList<Rect> separateEyes = new ArrayList<Rect>();
					if (leftEyes != null && leftEyes.length > 0) {
						separateEyes.add(leftEyes[0]);
					}
					if (rightEyes != null && rightEyes.length > 0) {
						separateEyes.add(rightEyes[0]);
					}
					eyes = separateEyes.toArray(new Rect[separateEyes.size()]);
				} else {
					Rect foundFaceForEyes = foundFaceInDetection.clone();
					eyes = cascadeEyesDetector.findEyes(mGray, foundFaceForEyes, false);
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

			if (eyes != null && eyes.length > 0) {
				ArrayList<Mat> eyesToShowAndProcess = new ArrayList<Mat>();
				Mat firstEyeToShow = new Mat(mGray, eyes[0]);
				int firstEyeRowStart = (int) (0.3 * mGray.height());
				eyesToShowAndProcess.add(firstEyeToShow);
				if (eyes.length == 2) {
					Mat secondEyeToShow = new Mat(mGray, eyes[1]);
					eyesToShowAndProcess.add(secondEyeToShow);
				}
				for (Mat eyeToShowAndProcess : eyesToShowAndProcess) {
					if (!isDoNothing) {
						if (isCannyAlgorithmUsed) {
							claheAlgorithm.process(eyeToShowAndProcess);
							edgeDetectionAlgorithm.cannyEdgeDetection(eyeToShowAndProcess);
						} else if (isSobelAlgorithmUsed){
							claheAlgorithm.process(eyeToShowAndProcess);
							Imgproc.GaussianBlur(eyeToShowAndProcess, eyeToShowAndProcess, new Size(gaussianBlur, gaussianBlur), 0);
							edgeDetectionAlgorithm.laplacianEdgeDetection(eyeToShowAndProcess);
							Scalar meanValues = Core.mean(eyeToShowAndProcess);
							darkBrightRatioAlgorithm.getBinarizationAlgorithm().setThreshold((int) meanValues.val[0]);
							//darkBrightRatioAlgorithm.getBinarizationAlgorithm().standardBinarization(eyeToShowAndProcess);
							darkBrightRatioAlgorithm.getBinarizationAlgorithm().adaptiveMeanBinarization(eyeToShowAndProcess);
						} else {
							if (isSimpleBinarizationUsed) {
								Scalar meanValues = Core.mean(eyeToShowAndProcess);
								darkBrightRatioAlgorithm.getBinarizationAlgorithm().setThreshold((int) meanValues.val[0]);
								darkBrightRatioAlgorithm.claheEqualizeSimpleBinarizeAndCloseOperation(eyeToShowAndProcess);
							} else {
								darkBrightRatioAlgorithm.claheEqualizeAdaptiveBinarizeAndCloseOperation(eyeToShowAndProcess);
							}
						}
					} else {
						claheAlgorithm.process(eyeToShowAndProcess);
						// Imgproc.GaussianBlur(eyeToShowAndProcess,
						// eyeToShowAndProcess, new Size(gaussianBlur,
						// gaussianBlur), 0);
					}
					VisualUtils.resizeImage(eyeToShowAndProcess, 3);
				}
				firstEyeToShow.copyTo(mGray.submat(firstEyeRowStart, firstEyeRowStart + firstEyeToShow.height(), 0, firstEyeToShow.width()));
				if (eyes.length == 2) {
					Mat secondEyeToShow = eyesToShowAndProcess.get(1);
					secondEyeToShow.copyTo(mGray.submat(firstEyeRowStart, firstEyeRowStart + secondEyeToShow.height(), mGray.width()
							- secondEyeToShow.width(), mGray.width()));
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

		frameCounter++;

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

	public DarkBrightRatioAlgorithm getDarkBrightRatioAlgorithm() {
		return darkBrightRatioAlgorithm;
	}

	public void setDarkBrightRatioAlgorithm(DarkBrightRatioAlgorithm darkBrightRatioAlgorithm) {
		this.darkBrightRatioAlgorithm = darkBrightRatioAlgorithm;
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

	public EdgeDetectionAlgorithm getEdgeDetectionAlgorithm() {
		return edgeDetectionAlgorithm;
	}

	public void setEdgeDetectionAlgorithm(EdgeDetectionAlgorithm edgeDetectionAlgorithm) {
		this.edgeDetectionAlgorithm = edgeDetectionAlgorithm;
	}

	public boolean isCannyAlgorithmUsed() {
		return isCannyAlgorithmUsed;
	}

	public void setCannyAlgorithmUsed(boolean isCannyAlgorithmUsed) {
		this.isCannyAlgorithmUsed = isCannyAlgorithmUsed;
	}

	public boolean isSobelAlgorithmUsed() {
		return isSobelAlgorithmUsed;
	}

	public void setSobelAlgorithmUsed(boolean isSobelAlgorithmUsed) {
		this.isSobelAlgorithmUsed = isSobelAlgorithmUsed;
	}

	public boolean isSimpleBinarizationUsed() {
		return isSimpleBinarizationUsed;
	}

	public void setSimpleBinarizationUsed(boolean isSimpleBinarizationUsed) {
		this.isSimpleBinarizationUsed = isSimpleBinarizationUsed;
	}

	public boolean isDoNothing() {
		return isDoNothing;
	}

	public void setDoNothing(boolean isDoNothing) {
		this.isDoNothing = isDoNothing;
	}

}
