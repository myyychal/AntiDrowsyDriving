package com.mpanek.detection.main;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.media.AudioManager;
import android.media.Ringtone;
import android.media.RingtoneManager;
import android.media.ToneGenerator;
import android.net.Uri;
import android.util.Log;

import com.mpanek.activities.main.MainActivity;
import com.mpanek.algorithms.general.ClaheAlgorithm;
import com.mpanek.algorithms.general.EdgeDetectionAlgorithm;
import com.mpanek.algorithms.general.HistogramEqualizationAlgorithm;
import com.mpanek.algorithms.specialized.DarkBrightRatioAlgorithm;
import com.mpanek.constants.DrawingConstants;
import com.mpanek.detection.elements.eyes.CascadeEyesDetector;
import com.mpanek.detection.elements.face.CascadeFaceDetector;
import com.mpanek.detection.elements.mouth.CascadeMouthDetector;
import com.mpanek.detection.elements.nose.CascadeNoseDetector;
import com.mpanek.utils.DrawingUtils;
import com.mpanek.utils.MathUtils;
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

	private boolean isLaplacianAlgorithmUsed = false;
	private boolean isSimpleBinarizationUsed = false;
	private boolean isAdaptiveBinarizationUsed = false;

	private boolean isDoNothing = false;

	private boolean isProjectionAnalysis = false;

	private boolean isIntensityVPA = false;
	private boolean isMeanValuesVPA = false;
	private boolean isIntensityHPA = false;
	private boolean isMeanValuesHPA = false;

	int gaussianBlur = 5;

	long frameCounter = 0;
	boolean isFaceFound = false;
	boolean isLeftEyeFound = false;
	boolean isRightEyeFound = false;

	boolean isEyeClosedMeanHPA = false;
	boolean isEyeClosedMeanVPA = false;

	long prevFrameTime = 0;
	long summaryTimeOfClosedEyes = 0;
	int countOfConsecutiveFramesWithOpenedEyesAfterClosedEyes = 0;

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
		darkBrightRatioAlgorithm.getBinarizationAlgorithm().setBlockSize(45);
		darkBrightRatioAlgorithm.setErosionSize(5);
		this.edgeDetectionAlgorithm = edgeDetectionAlgorithm;
	}

	public Mat processDetection(Mat mGray, Mat mRgba) {
		isEyeClosedMeanHPA = false;
		isEyeClosedMeanVPA = false;
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
				// Imgproc.equalizeHist(imgToFindWithROI, imgToFindWithROI);
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
				ArrayList<Long> vpaValuesArrayList = null;
				for (Mat eyeToShowAndProcess : eyesToShowAndProcess) {
					if (!isProjectionAnalysis) {
						if (isLaplacianAlgorithmUsed) {
							claheAlgorithm.process(eyeToShowAndProcess);
							// Imgproc.equalizeHist(firstEyeToShow,
							// firstEyeToShow);
							int otherGaussianBlurSize = 2 * gaussianBlur + 1;
							if (otherGaussianBlurSize % 2 == 0) {
								otherGaussianBlurSize += 1;
							}
							Imgproc.GaussianBlur(eyeToShowAndProcess, eyeToShowAndProcess, new Size(otherGaussianBlurSize, otherGaussianBlurSize), 0);
							edgeDetectionAlgorithm.laplacianAdvancedEdgeDetection(eyeToShowAndProcess);
							Scalar meanValues = Core.mean(eyeToShowAndProcess);
							darkBrightRatioAlgorithm.getBinarizationAlgorithm().setThreshold((int) meanValues.val[0]);
							darkBrightRatioAlgorithm.getBinarizationAlgorithm().adaptiveMeanBinarization(eyeToShowAndProcess);
							darkBrightRatioAlgorithm.performOpenOperation(eyeToShowAndProcess, 1);
							darkBrightRatioAlgorithm.performErodeOperation(eyeToShowAndProcess, 1);
							darkBrightRatioAlgorithm.performCloseOperation(eyeToShowAndProcess, 2);
							// darkBrightRatioAlgorithm.performErodeOperation(eyeToShowAndProcess,
							// 1);
							int sideMargin = (int) (eyeToShowAndProcess.width() / 2.7);
							DrawingUtils.removeVerticalBorders(eyeToShowAndProcess, sideMargin, 0);
							DrawingUtils.removeHorizontalBorders(eyeToShowAndProcess, eyeToShowAndProcess.width() / 12, 0);
							Rect roi = new Rect(new Point(sideMargin, 0), new Point(eyeToShowAndProcess.width() - sideMargin,
									eyeToShowAndProcess.height()));
							Mat roiMat = new Mat(eyeToShowAndProcess, roi);
							byte buff[] = new byte[(int) (roiMat.total() * roiMat.channels())];
							roiMat.get(0, 0, buff);
							ArrayList<Long> sumOfWhitePixelsInRow = darkBrightRatioAlgorithm.countIntensityVerticalProjection(roiMat);
							int firstIndex = 0;
							int lastIndex = roiMat.width();
							long maxValue = roiMat.width() * 255;
							for (int i = 0; i < sumOfWhitePixelsInRow.size(); i++) {
								long value = sumOfWhitePixelsInRow.get(i);
								if (value >= maxValue / 4 && firstIndex == 0) {
									firstIndex = i;
								}
								if (value >= maxValue / 4) {
									lastIndex = i;
								}
							}
							DrawingUtils.drawLines(new Point[] { new Point(0, firstIndex), new Point(eyeToShowAndProcess.width(), firstIndex) },
									eyeToShowAndProcess, DrawingConstants.WHITE);
							DrawingUtils.drawLines(new Point[] { new Point(0, lastIndex), new Point(eyeToShowAndProcess.width(), lastIndex) },
									eyeToShowAndProcess, DrawingConstants.WHITE);
							roiMat.put(0, 0, buff);
						} else if (isSimpleBinarizationUsed) {
							int sideMargin = (int) (firstEyeToShow.width() / 3.5);
							DrawingUtils.removeVerticalBorders(firstEyeToShow, sideMargin, 0);
							Rect roi = new Rect(new Point(sideMargin, 0), new Point(firstEyeToShow.width() - sideMargin, firstEyeToShow.height()));
							Mat roiMat = new Mat(firstEyeToShow, roi);

							Scalar meanValues = Core.mean(roiMat);
							darkBrightRatioAlgorithm.getBinarizationAlgorithm().setThreshold((int) meanValues.val[0]);
							darkBrightRatioAlgorithm.claheEqualizeSimpleBinarizeAndOpenOperation(roiMat);
							// darkBrightRatioAlgorithm.performDilateOperation(roiMat,
							// 3);
							// DrawingUtils.removeAllBorders(roiMat, 8, 255);
							darkBrightRatioAlgorithm.countMeanBlackAndWhitePixels(roiMat);
						} else if (isAdaptiveBinarizationUsed) {
							int sideMargin = (int) (firstEyeToShow.width() / 3.5);
							DrawingUtils.removeVerticalBorders(firstEyeToShow, sideMargin, 0);
							Rect roi = new Rect(new Point(sideMargin, 0), new Point(firstEyeToShow.width() - sideMargin, firstEyeToShow.height()));
							Mat roiMat = new Mat(firstEyeToShow, roi);

							darkBrightRatioAlgorithm.claheEqualizeAdaptiveBinarizeAndOpenOperation(roiMat);
							darkBrightRatioAlgorithm.performDilateOperation(roiMat, 3);
							// DrawingUtils.removeAllBorders(roiMat, 8, 255);
							darkBrightRatioAlgorithm.countMeanBlackAndWhitePixels(roiMat);
						} else if (isDoNothing) {
							darkBrightRatioAlgorithm.countMeanBlackAndWhitePixels(eyesToShowAndProcess.get(0));
						}
					}
					VisualUtils.resizeImage(eyeToShowAndProcess, 3);
				}
				if (isProjectionAnalysis) {
					claheAlgorithm.process(firstEyeToShow);
					// Imgproc.equalizeHist(firstEyeToShow, firstEyeToShow);
					Imgproc.GaussianBlur(firstEyeToShow, firstEyeToShow, new Size(9, 9), 0);
					int sideMargin = (int) (firstEyeToShow.width() / 3.5);
					DrawingUtils.removeVerticalBorders(firstEyeToShow, sideMargin, 0);
					Rect roi = new Rect(new Point(sideMargin, 0), new Point(firstEyeToShow.width() - sideMargin, firstEyeToShow.height()));
					Mat roiMat = new Mat(firstEyeToShow, roi);
					if (isIntensityVPA) {
						vpaValuesArrayList = darkBrightRatioAlgorithm.countIntensityVerticalProjection(roiMat);
						vpaValuesArrayList = MathUtils.applyMedianFilterOnLongArray(vpaValuesArrayList, 5);
						darkBrightRatioAlgorithm.normalizeAndDrawVerticalProjectionAnalysisArrays(roiMat, vpaValuesArrayList);
					} else if (isMeanValuesVPA) {
						darkBrightRatioAlgorithm.fillWhiteSpots(firstEyeToShow, 180, 255, firstEyeToShow.height() / 4);
						vpaValuesArrayList = darkBrightRatioAlgorithm.countMeanVerticalProjection(roiMat);
						vpaValuesArrayList = MathUtils.applyMedianFilterOnLongArray(vpaValuesArrayList, 5);
						ArrayList<Long> normalizedList = darkBrightRatioAlgorithm.normalizeAndDrawVerticalProjectionAnalysisArrays(roiMat,
								vpaValuesArrayList, 0, 255);
						final long min = MathUtils.findMin(normalizedList);
						final long max = MathUtils.findMax(normalizedList);
						int aboveValuesCounter = 0;
						for (Long value : normalizedList) {
							if (value <= min + (max - min) / 4) {
								aboveValuesCounter++;
							}
						}
						int currentAboveCounterPercentage = aboveValuesCounter * 100 / normalizedList.size();
						String currentAboveValuesCounterPercentageString = String.valueOf(currentAboveCounterPercentage).concat(" %");
						DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(0, firstEyeRowStart - 25));
						DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(0, firstEyeRowStart - 26));
						DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(1, firstEyeRowStart - 25));
						if (currentAboveCounterPercentage < 30) {
							isEyeClosedMeanVPA = true;
							final ToneGenerator tg = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100);
							tg.startTone(ToneGenerator.TONE_PROP_BEEP);
						}
					} else if (isIntensityHPA) {
						vpaValuesArrayList = darkBrightRatioAlgorithm.countIntensityHorizontalProjection(roiMat);
						vpaValuesArrayList = MathUtils.applyMedianFilterOnLongArray(vpaValuesArrayList, 5);
						darkBrightRatioAlgorithm.normalizeAndDrawHorizontalProjectionAnalysisArrays(roiMat, vpaValuesArrayList);
					} else if (isMeanValuesHPA) {
						darkBrightRatioAlgorithm.fillWhiteSpots(firstEyeToShow, 180, 255, firstEyeToShow.height() / 4);
						vpaValuesArrayList = darkBrightRatioAlgorithm.countMeanHorizontalProjection(roiMat);
						vpaValuesArrayList = MathUtils.applyMedianFilterOnLongArray(vpaValuesArrayList, 5);
						ArrayList<Long> normalizedList = darkBrightRatioAlgorithm.normalizeAndDrawHorizontalProjectionAnalysisArrays(roiMat,
								vpaValuesArrayList, 0, 255);
						long firstVal = normalizedList.get(0);
						long lastVal = normalizedList.get(normalizedList.size() - 1);
						long diff = lastVal - firstVal;
						DrawingUtils.drawLines(new Point[] { new Point(0, firstVal), new Point(roiMat.width() - 1, lastVal) }, roiMat,
								DrawingConstants.WHITE);
						ArrayList<Long> simpleLineValues = new ArrayList<Long>();
						for (int i = 0; i < normalizedList.size(); i++) {
							simpleLineValues.add(normalizedList.get(0) + i * diff / normalizedList.size());
						}
						int aboveLineCounter = 0;
						for (int i = 0; i < normalizedList.size(); i++) {
							if (normalizedList.get(i) < simpleLineValues.get(i)) {
								aboveLineCounter++;
							}
						}
						int currentAboveCounterPercentage = aboveLineCounter * 100 / normalizedList.size();
						String currentAboveValuesCounterPercentageString = String.valueOf(currentAboveCounterPercentage).concat(" %");
						DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(0, firstEyeRowStart - 25));
						DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(0, firstEyeRowStart - 26));
						DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(1, firstEyeRowStart - 25));
						if (currentAboveCounterPercentage < 50) {
							isEyeClosedMeanHPA = true;
							final ToneGenerator tg = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100);
							tg.startTone(ToneGenerator.TONE_PROP_BEEP);
						}
					}

				}
				if (isDoNothing) {
					claheAlgorithm.process(firstEyeToShow);
					darkBrightRatioAlgorithm.fillWhiteSpots(firstEyeToShow, 180, 255, firstEyeToShow.height() / 4);
				}
				darkBrightRatioAlgorithm.countMeanBlackAndWhitePixels(firstEyeToShow);
				firstEyeToShow.copyTo(mGray.submat(firstEyeRowStart, firstEyeRowStart + firstEyeToShow.height(), 0, firstEyeToShow.width()));
				if (eyes.length == 2) {
					Mat secondEyeToShow = eyesToShowAndProcess.get(1);
					secondEyeToShow.copyTo(mGray.submat(firstEyeRowStart, firstEyeRowStart + secondEyeToShow.height(), mGray.width()
							- secondEyeToShow.width(), mGray.width()));
				}
				DrawingUtils.putText(mGray, "meanValue: " + String.format("%.2f", darkBrightRatioAlgorithm.getMeanValuePixels()), new Point(0,
						firstEyeRowStart + firstEyeToShow.height() + 20));
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

	public Mat processDetectionFinal(Mat mGray, Mat mRgba) {
		isEyeClosedMeanHPA = false;
		isEyeClosedMeanVPA = false;

		claheAlgorithm.process(mGray);
		Imgproc.GaussianBlur(mGray, mGray, new Size(gaussianBlur, gaussianBlur), 0);

		Rect foundFaceInDetection = new Rect(0, 0, mGray.width(), mGray.height());

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

		if (foundFaceInDetection != null) {
			Rect[] eyes = null;

			Mat imgToFindWithROI;
			try {
				imgToFindWithROI = new Mat(mGray, foundFaceInDetection);
			} catch (CvException e) {
				return mRgba;
			}

			claheAlgorithm.process(imgToFindWithROI);
			Imgproc.GaussianBlur(mGray, mGray, new Size(gaussianBlur, gaussianBlur), 0);

			Rect foundFaceForEyes = foundFaceInDetection.clone();
			eyes = cascadeEyesDetector.findEyes(mGray, foundFaceForEyes, false);
			if (eyes == null || eyes.length == 0) {
				eyes = cascadeEyesDetector.getLastFoundEyes();
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
				ArrayList<Long> vpaValuesArrayList = null;

				VisualUtils.resizeImage(firstEyeToShow, 3);

				claheAlgorithm.process(firstEyeToShow);
				// Imgproc.equalizeHist(firstEyeToShow, firstEyeToShow);
				Imgproc.GaussianBlur(firstEyeToShow, firstEyeToShow, new Size(9, 9), 0);
				int sideMargin = (int) (firstEyeToShow.width() / 3.5);
				DrawingUtils.removeVerticalBorders(firstEyeToShow, sideMargin, 0);
				Rect roi = new Rect(new Point(sideMargin, 0), new Point(firstEyeToShow.width() - sideMargin, firstEyeToShow.height()));
				Mat roiMat = new Mat(firstEyeToShow, roi);
				if (isMeanValuesVPA) {
					darkBrightRatioAlgorithm.fillWhiteSpots(firstEyeToShow, 180, 255, firstEyeToShow.height() / 4);
					vpaValuesArrayList = darkBrightRatioAlgorithm.countMeanVerticalProjection(roiMat);
					vpaValuesArrayList = MathUtils.applyMedianFilterOnLongArray(vpaValuesArrayList, 5);
					ArrayList<Long> normalizedList = darkBrightRatioAlgorithm.normalizeAndDrawVerticalProjectionAnalysisArrays(roiMat,
							vpaValuesArrayList, 0, 255);
					final long min = MathUtils.findMin(normalizedList);
					final long max = MathUtils.findMax(normalizedList);
					int aboveValuesCounter = 0;
					for (Long value : normalizedList) {
						if (value <= min + (max - min) / 4) {
							aboveValuesCounter++;
						}
					}
					int currentAboveCounterPercentage = aboveValuesCounter * 100 / normalizedList.size();
					String currentAboveValuesCounterPercentageString = String.valueOf(currentAboveCounterPercentage).concat(" %");
					DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(0, firstEyeRowStart - 25));
					DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(0, firstEyeRowStart - 26));
					DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(1, firstEyeRowStart - 25));
					if (currentAboveCounterPercentage < 30) {
						isEyeClosedMeanVPA = true;
						final ToneGenerator tg = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100);
						tg.startTone(ToneGenerator.TONE_PROP_BEEP);
					}
				} else if (isMeanValuesHPA) {
					darkBrightRatioAlgorithm.fillWhiteSpots(firstEyeToShow, 180, 255, firstEyeToShow.height() / 4);
					vpaValuesArrayList = darkBrightRatioAlgorithm.countMeanHorizontalProjection(roiMat);
					vpaValuesArrayList = MathUtils.applyMedianFilterOnLongArray(vpaValuesArrayList, 5);
					ArrayList<Long> normalizedList = darkBrightRatioAlgorithm.normalizeAndDrawHorizontalProjectionAnalysisArrays(roiMat,
							vpaValuesArrayList, 0, 255);
					long firstVal = normalizedList.get(0);
					long lastVal = normalizedList.get(normalizedList.size() - 1);
					long diff = lastVal - firstVal;
					DrawingUtils.drawLines(new Point[] { new Point(0, firstVal), new Point(roiMat.width() - 1, lastVal) }, roiMat,
							DrawingConstants.WHITE);
					ArrayList<Long> simpleLineValues = new ArrayList<Long>();
					for (int i = 0; i < normalizedList.size(); i++) {
						simpleLineValues.add(normalizedList.get(0) + i * diff / normalizedList.size());
					}
					int aboveLineCounter = 0;
					for (int i = 0; i < normalizedList.size(); i++) {
						if (normalizedList.get(i) < simpleLineValues.get(i)) {
							aboveLineCounter++;
						}
					}
					int currentAboveCounterPercentage = aboveLineCounter * 100 / normalizedList.size();
					String currentAboveValuesCounterPercentageString = String.valueOf(currentAboveCounterPercentage).concat(" %");
					DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(0, firstEyeRowStart - 25));
					DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(0, firstEyeRowStart - 26));
					DrawingUtils.putText(mGray, currentAboveValuesCounterPercentageString, new Point(1, firstEyeRowStart - 25));
					if (currentAboveCounterPercentage < 50) {
						isEyeClosedMeanHPA = true;
						final ToneGenerator tg = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100);
						tg.startTone(ToneGenerator.TONE_PROP_BEEP);
					}
				}

				darkBrightRatioAlgorithm.countMeanBlackAndWhitePixels(firstEyeToShow);
				firstEyeToShow.copyTo(mGray.submat(firstEyeRowStart, firstEyeRowStart + firstEyeToShow.height(), 0, firstEyeToShow.width()));
				DrawingUtils.putText(mGray, "meanValue: " + String.format("%.2f", darkBrightRatioAlgorithm.getMeanValuePixels()), new Point(0,
						firstEyeRowStart + firstEyeToShow.height() + 20));
			}

			Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA);

			DrawingUtils.drawRect(foundFaceInDetection, mRgba, DrawingConstants.FACE_RECT_COLOR);

			DrawingUtils.drawRects(eyes, mRgba, DrawingConstants.EYES_RECT_COLOR);

			ArrayList<Rect> allDetectedRects = new ArrayList<Rect>();
			if (foundFaceInDetection != null) {
				allDetectedRects.add(foundFaceInDetection);
			}
			if (eyes != null) {
				for (Rect eyeRect : eyes) {
					if (eyeRect != null) {
						allDetectedRects.add(eyeRect);
					}
				}
			}

			DrawingUtils.drawLinesFromRectanglesCentres(allDetectedRects.toArray(new Rect[allDetectedRects.size()]), mRgba, DrawingConstants.WHITE);

		} else {
			Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA);
		}

		frameCounter++;

		return mRgba;
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

	public void setAllClosedEyeDetectionMethods(boolean value) {
		isLaplacianAlgorithmUsed = value;
		isSimpleBinarizationUsed = value;
		isAdaptiveBinarizationUsed = value;
		isDoNothing = value;
		isProjectionAnalysis = value;
		isIntensityVPA = value;
		isMeanValuesVPA = value;
		isIntensityHPA = value;
		isMeanValuesHPA = value;
	}

	public void horizontalLineDetection(Mat frame, int heightOfElement) {
		int horizontalSize = darkBrightRatioAlgorithm.getErosionSize();
		Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * horizontalSize / 2 + 1, heightOfElement));
		Imgproc.erode(frame, frame, horizontalStructure);
		Imgproc.dilate(frame, frame, horizontalStructure);
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

	public boolean isLaplacianAlgorithmUsed() {
		return isLaplacianAlgorithmUsed;
	}

	public void setLaplacianAlgorithmUsed(boolean isLaplacianAlgorithmUsed) {
		this.isLaplacianAlgorithmUsed = isLaplacianAlgorithmUsed;
	}

	public boolean isSimpleBinarizationUsed() {
		return isSimpleBinarizationUsed;
	}

	public void setSimpleBinarizationUsed(boolean isSimpleBinarizationUsed) {
		this.isSimpleBinarizationUsed = isSimpleBinarizationUsed;
	}

	public boolean isAdaptiveBinarizationUsed() {
		return isAdaptiveBinarizationUsed;
	}

	public void setAdaptiveBinarizationUsed(boolean isAdaptiveBinarizationUsed) {
		this.isAdaptiveBinarizationUsed = isAdaptiveBinarizationUsed;
	}

	public boolean isProjectionAnalysis() {
		return isProjectionAnalysis;
	}

	public void setProjectionAnalysis(boolean isDoNothing) {
		this.isProjectionAnalysis = isDoNothing;
	}

	public boolean isIntensityVPA() {
		return isIntensityVPA;
	}

	public void setIntensityVPA(boolean isIntensityVPA) {
		this.isIntensityVPA = isIntensityVPA;
	}

	public boolean isMeanValuesVPA() {
		return isMeanValuesVPA;
	}

	public void setMeanValuesVPA(boolean isMeanValuesVPA) {
		this.isMeanValuesVPA = isMeanValuesVPA;
	}

	public boolean isDoNothing() {
		return isDoNothing;
	}

	public void setDoNothing(boolean isDoNothing) {
		this.isDoNothing = isDoNothing;
	}

	public boolean isIntensityHPA() {
		return isIntensityHPA;
	}

	public void setIntensityHPA(boolean isIntensityHPA) {
		this.isIntensityHPA = isIntensityHPA;
	}

	public boolean isMeanValuesHPA() {
		return isMeanValuesHPA;
	}

	public void setMeanValuesHPA(boolean isMeanValuesHPA) {
		this.isMeanValuesHPA = isMeanValuesHPA;
	}

}
