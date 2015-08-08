package com.mpanek.activities.main;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.Objdetect;
import org.opencv.video.Video;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.DialogInterface.OnClickListener;
import android.hardware.SensorManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.OrientationEventListener;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.PopupMenu;
import android.widget.PopupMenu.OnMenuItemClickListener;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VerticalSeekBar;

import com.mpanek.algorithms.general.BinarizationAlgorithm;
import com.mpanek.algorithms.general.ClaheAlgorithm;
import com.mpanek.algorithms.general.EdgeDetectionAlgorithm;
import com.mpanek.algorithms.general.HistogramEqualizationAlgorithm;
import com.mpanek.algorithms.specialized.DarkBrightRatioAlgorithm;
import com.mpanek.constants.DetectorConstants;
import com.mpanek.constants.DrawingConstants;
import com.mpanek.constants.ViewModesConstants;
import com.mpanek.detection.elements.SnapdragonFacialFeaturesDetector;
import com.mpanek.detection.elements.eyes.CascadeEyesDetector;
import com.mpanek.detection.elements.face.CascadeFaceDetector;
import com.mpanek.detection.elements.face.ColorSegmentationFaceDetector;
import com.mpanek.detection.elements.mouth.CascadeMouthDetector;
import com.mpanek.detection.elements.nose.CascadeNoseDetector;
import com.mpanek.detection.main.DrowsinessDetector;
import com.mpanek.tasks.FaceDetectionAsyncTask;
import com.mpanek.tasks.GaussBlurAsyncTask;
import com.mpanek.utils.DrawingUtils;
import com.mpanek.utils.MathUtils;
import com.mpanek.utils.VisualUtils;
import com.mpanek.views.camera.CustomCameraView;

public class MainActivity extends Activity implements CvCameraViewListener2 {
	private static final String TAG = "AntiDrowsyDriving::MainActivity";

	TextView scaleFactorValueText;
	TextView minNeighsValueText;
	TextView minFaceValueText;
	TextView maxFaceValueText;
	TextView minEyeValueText;
	TextView maxEyeValueText;
	TextView clipLimitValueText;
	TextView tileSizeValueText;
	TextView minObjectValueText;
	TextView maxObjectValueText;
	TextView gaussValueText;
	TextView blockSizeText;
	TextView cText;
	TextView thresholdText;
	TextView firstThresholdText;
	TextView secondThresholdText;
	TextView apertureSizeText;
	TextView erosionSizeText;

	private VerticalSeekBar scaleFactorSeekBar;
	private VerticalSeekBar minNeighsSeekBar;
	private VerticalSeekBar minFaceSeekBar;
	private VerticalSeekBar maxFaceSeekBar;
	private VerticalSeekBar minEyeSeekBar;
	private VerticalSeekBar maxEyeSeekBar;
	private VerticalSeekBar minObjectSeekBar;
	private VerticalSeekBar maxObjectSeekBar;
	private VerticalSeekBar clipLimitSeekBar;
	private VerticalSeekBar tileSizeSeekBar;
	private VerticalSeekBar gaussSeekBar;
	private VerticalSeekBar blockSizeSeekBar;
	private VerticalSeekBar cSeekBar;
	private VerticalSeekBar thresholdSeekBar;
	private VerticalSeekBar firstThresholdSeekBar;
	private VerticalSeekBar secondThresholdSeekBar;
	private VerticalSeekBar apertureSizeSeekBar;
	private VerticalSeekBar erosionSizeSeekBar;

	Map<String, VerticalSeekBar> verticalSeekBars = new LinkedHashMap<String, VerticalSeekBar>();
	Map<String, TextView> verticalSeekBarsTextValues = new LinkedHashMap<String, TextView>();
	Map<String, TextView> verticalSeekBarsTextNames = new LinkedHashMap<String, TextView>();

	private CheckBox gaussCheckbox;

	private int mCameraId = CameraBridgeViewBase.CAMERA_ID_FRONT;

	private int mViewMode;
	private int mEqHistMode;

	private Mat mRgba;
	private Mat mGray;
	private Mat currentlyUsedFrame;

	private Mat matOpFlowPrev, matOpFlowThis;
	private MatOfPoint MOPcorners;
	private MatOfPoint2f mMOP2f1, mMOP2f2, mMOP2fptsPrev, mMOP2fptsThis, mMOP2fptsSafe;
	private MatOfByte mMOBStatus;
	private MatOfFloat mMOFerr;
	private List<Point> cornersThis, cornersPrev;
	private List<Byte> byteStatus;
	private int iGFFTMax = 40;

	private List<android.hardware.Camera.Size> mResolutionList;
	private MenuItem[] mResolutionMenuItems;

	private CustomCameraView mCustomCameraView;

	BinarizationAlgorithm binarizationAlgorithm;
	ClaheAlgorithm claheAlgorithm;
	HistogramEqualizationAlgorithm histogramEqualizationAlgorithm;
	EdgeDetectionAlgorithm edgeDetectionAlgorithm;
	DarkBrightRatioAlgorithm darkBrightRatioAlgorithm;

	SnapdragonFacialFeaturesDetector snapdragonFacialFeaturesDetector;
	CascadeFaceDetector cascadeFaceDetector;
	CascadeEyesDetector cascadeEyesDetector;
	CascadeMouthDetector cascadeMouthDetector;
	CascadeNoseDetector cascadeNoseDetector;
	ColorSegmentationFaceDetector colorSegmentationFaceDetector;

	CascadeEyesDetector cascadeLeftEyeDetector;
	CascadeEyesDetector cascadeRightEyeDetector;

	DrowsinessDetector drowsinessDetector;

	OrientationEventListener orientationEventListener;
	int deviceOrientation;

	private static boolean isMouthWithEyes = true;
	private static boolean isFaceTracking = true;
	private static boolean isCurrentFrameRgb = true;

	private static boolean isGuiHidden = false;
	private static boolean isJavaCamera = true;

	private static boolean isNoseChosen = false;
	private static boolean isMouthChosen = false;

	private int gaussSize = 1;

	GaussBlurAsyncTask gaussBlurAsyncTask = new GaussBlurAsyncTask(currentlyUsedFrame, gaussSize);
	FaceDetectionAsyncTask faceDetectionAsyncTask = new FaceDetectionAsyncTask(currentlyUsedFrame, cascadeFaceDetector);

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");

				System.loadLibrary("anti_drowsy_driving");
				int faceResourceId = getResources().getIdentifier("lbpcascade_frontalface", "raw", getPackageName());
				InputStream is = getResources().openRawResource(faceResourceId);
				File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
				File faceFile = cascadeFaceDetector.prepare(is, cascadeDir);
				PrepareFindFace(faceFile.getAbsolutePath());

				int eyesResourceId = getResources().getIdentifier("haarcascade_righteye_2splits", "raw", getPackageName());
				is = getResources().openRawResource(eyesResourceId);
				File eyesFile = cascadeEyesDetector.prepare(is, cascadeDir);
				PrepareFindEyes(eyesFile.getAbsolutePath());

				eyesResourceId = getResources().getIdentifier("haarcascade_mcs_lefteye", "raw", getPackageName());
				is = getResources().openRawResource(eyesResourceId);
				eyesFile = cascadeLeftEyeDetector.prepare(is, cascadeDir);

				eyesResourceId = getResources().getIdentifier("haarcascade_mcs_righteye", "raw", getPackageName());
				is = getResources().openRawResource(eyesResourceId);
				eyesFile = cascadeRightEyeDetector.prepare(is, cascadeDir);

				int mouthResourceId = getResources().getIdentifier("haarcascade_mcs_mouth", "raw", getPackageName());
				is = getResources().openRawResource(mouthResourceId);
				cascadeMouthDetector.prepare(is, cascadeDir);

				int noseResourceId = getResources().getIdentifier("haarcascade_mcs_nose", "raw", getPackageName());
				is = getResources().openRawResource(noseResourceId);
				cascadeNoseDetector.prepare(is, cascadeDir);

				mCustomCameraView.enableView();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	public MainActivity() {
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		switch (event.getAction()) {
		case MotionEvent.ACTION_DOWN:
			if (mCameraId == CameraBridgeViewBase.CAMERA_ID_FRONT) {
				mCameraId = CameraBridgeViewBase.CAMERA_ID_BACK;
			} else {
				mCameraId = CameraBridgeViewBase.CAMERA_ID_FRONT;
			}

			mCustomCameraView.setVisibility(SurfaceView.GONE);
			mCustomCameraView.setCameraIndex(mCameraId);
			mCustomCameraView.setVisibility(SurfaceView.VISIBLE);
			mCustomCameraView.setCvCameraViewListener(this);
			mCustomCameraView.enableView();
		}
		return super.onTouchEvent(event);

	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.main_surface_view);

		mCustomCameraView = (CustomCameraView) findViewById(R.id.custom_camera_view);
		mCustomCameraView.setCameraIndex(mCameraId);
		mCustomCameraView.setVisibility(SurfaceView.VISIBLE);
		mCustomCameraView.setCvCameraViewListener(this);

		startOrientationListener();

		binarizationAlgorithm = new BinarizationAlgorithm();
		claheAlgorithm = new ClaheAlgorithm();
		histogramEqualizationAlgorithm = new HistogramEqualizationAlgorithm();
		edgeDetectionAlgorithm = new EdgeDetectionAlgorithm();
		darkBrightRatioAlgorithm = new DarkBrightRatioAlgorithm(claheAlgorithm, histogramEqualizationAlgorithm, binarizationAlgorithm);

		snapdragonFacialFeaturesDetector = new SnapdragonFacialFeaturesDetector(
				((WindowManager) getSystemService(Context.WINDOW_SERVICE)).getDefaultDisplay(), mCameraId);

		cascadeFaceDetector = new CascadeFaceDetector();
		cascadeEyesDetector = new CascadeEyesDetector();
		cascadeMouthDetector = new CascadeMouthDetector();
		cascadeNoseDetector = new CascadeNoseDetector();

		cascadeLeftEyeDetector = new CascadeEyesDetector();
		cascadeLeftEyeDetector.setCascadeFileName("haarcascade_mcs_lefteye");
		cascadeRightEyeDetector = new CascadeEyesDetector();
		cascadeRightEyeDetector.setCascadeFileName("haarcascade_mcs_righteye");

		colorSegmentationFaceDetector = new ColorSegmentationFaceDetector();

		drowsinessDetector = new DrowsinessDetector(cascadeFaceDetector, cascadeEyesDetector, cascadeMouthDetector, cascadeNoseDetector,
				claheAlgorithm, darkBrightRatioAlgorithm, edgeDetectionAlgorithm);
		drowsinessDetector.setCascadeLeftEyeDetector(cascadeLeftEyeDetector);
		drowsinessDetector.setCascadeRightEyeDetector(cascadeRightEyeDetector);
		drowsinessDetector.setSeparateEyesDetection(false);
		drowsinessDetector.setCannyAlgorithmUsed(false);

		initVerticalSeekBars();

		gaussCheckbox = (CheckBox) findViewById(R.id.gaussCheckbox);

		isRawDataExists(this);
	}

	private void startOrientationListener() {
		orientationEventListener = new OrientationEventListener(this, SensorManager.SENSOR_DELAY_NORMAL) {
			@Override
			public void onOrientationChanged(int orientation) {
				deviceOrientation = orientation;
			}
		};

		if (orientationEventListener.canDetectOrientation()) {
			orientationEventListener.enable();
		}

	}

	@Override
	public void onPause() {
		super.onPause();
		if (mCustomCameraView != null)
			mCustomCameraView.disableView();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		if (mCustomCameraView != null)
			mCustomCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mRgba = new Mat(height, width, CvType.CV_8UC4);
		currentlyUsedFrame = new Mat();
		mGray = new Mat(height, width, CvType.CV_8UC1);

		matOpFlowThis = new Mat();
		matOpFlowPrev = new Mat();

		MOPcorners = new MatOfPoint();

		mMOBStatus = new MatOfByte();

		mMOFerr = new MatOfFloat();

		mMOP2f1 = new MatOfPoint2f();
		mMOP2f2 = new MatOfPoint2f();
		mMOP2fptsPrev = new MatOfPoint2f();
		mMOP2fptsThis = new MatOfPoint2f();
		mMOP2fptsSafe = new MatOfPoint2f();

		byteStatus = new ArrayList<Byte>();

		cornersThis = new ArrayList<Point>();
		cornersPrev = new ArrayList<Point>();

		// android.hardware.Camera.Size resolution =
		// mCustomCameraView.getCamera().new Size(640, 480);
		// mCustomCameraView.setResolution(resolution);
	}

	public void onCameraViewStopped() {
		mRgba.release();
		mGray.release();
		currentlyUsedFrame.release();

		MOPcorners.release();

		mMOBStatus.release();

		mMOFerr.release();

		mMOP2f1.release();
		mMOP2f2.release();
		mMOP2fptsPrev.release();
		mMOP2fptsThis.release();
		mMOP2fptsSafe.release();
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();

		if (mCameraId == CameraBridgeViewBase.CAMERA_ID_FRONT) {
			// if commented then cascadeLeft and cascadeRight should be used for
			// opposite eyes
			// Core.flip(mRgba, mRgba, 1);
			// Core.flip(mGray, mGray, 1);
		}

		if (isCurrentFrameRgb) {
			currentlyUsedFrame = mRgba;
		} else {
			currentlyUsedFrame = mGray;
		}

		final int viewMode = mViewMode;
		final int eqHistMode = mEqHistMode;

		if (gaussCheckbox.isChecked() && viewMode != ViewModesConstants.VIEW_MODE_START_DROWSINESS_DETECTION) {
			// Imgproc.GaussianBlur(currentlyUsedFrame, currentlyUsedFrame,
			// new Size(gaussSize, gaussSize), 0);
			AsyncTask.Status gaussBlurSyncTaskStatus = gaussBlurAsyncTask.getStatus();
			if (gaussBlurSyncTaskStatus.equals(AsyncTask.Status.FINISHED)) {
				Log.i(TAG, "gaussBlurAsyncTask is finished");
				try {
					currentlyUsedFrame = gaussBlurAsyncTask.get();
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
				gaussBlurAsyncTask = new GaussBlurAsyncTask(currentlyUsedFrame, gaussSize);
				gaussBlurAsyncTask.execute();
			} else if (gaussBlurSyncTaskStatus.equals(AsyncTask.Status.PENDING)) {
				Log.i(TAG, "gaussBlurAsyncTask is pending");
				gaussBlurAsyncTask.setFrame(currentlyUsedFrame);
				gaussBlurAsyncTask.execute();
			} else if (gaussBlurSyncTaskStatus.equals(AsyncTask.Status.RUNNING)) {
				Log.i(TAG, "gaussBlurAsyncTask is running");
			}
		}

		switch (eqHistMode) {
		case ViewModesConstants.VIEW_MODE_EQ_NONE:
			break;

		case ViewModesConstants.VIEW_MODE_EQ_HIST_CPP:
			histogramEqualizationAlgorithm.standardEqualizationCpp(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_EQ_HIST_JAVA:
			histogramEqualizationAlgorithm.standardEqualizationJava(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_EQ_HIST_CLAHE_CPP:
			claheAlgorithm.process(currentlyUsedFrame);
			break;
		}

		switch (viewMode) {

		case ViewModesConstants.VIEW_MODE_NONE:
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_SNAPDRAGON:
			snapdragonFacialFeaturesDetector.findFace(mRgba, this.getResources().getConfiguration().orientation);
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_JAVA:
			Rect face = cascadeFaceDetector.findFace(currentlyUsedFrame);
			if (face != null) {
				DrawingUtils.drawRect(face, currentlyUsedFrame, DrawingConstants.FACE_RECT_COLOR);
				// Mat imgToFindWithROI = new Mat(currentlyUsedFrame, face);
				// imgToFindWithROI.copyTo(currentlyUsedFrame.submat(0,
				// imgToFindWithROI.height(), 0, imgToFindWithROI.width()));
			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_CPP:
			FindFace(currentlyUsedFrame.getNativeObjAddr(), currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_JAVA_ASYNC_TASK:
			Rect foundFaceFromAsyncTask = null;
			AsyncTask.Status faceDetectSyncTaskStatus = faceDetectionAsyncTask.getStatus();
			if (faceDetectSyncTaskStatus.equals(AsyncTask.Status.FINISHED)) {
				Log.i(TAG, "faceDetectAsyncTask is finished");
				try {
					foundFaceFromAsyncTask = faceDetectionAsyncTask.get();
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
				faceDetectionAsyncTask = new FaceDetectionAsyncTask(currentlyUsedFrame, cascadeFaceDetector);
				faceDetectionAsyncTask.execute();
			} else if (faceDetectSyncTaskStatus.equals(AsyncTask.Status.PENDING)) {
				Log.i(TAG, "faceDetectAsyncTask is pending");
				faceDetectionAsyncTask.setFrame(currentlyUsedFrame);
				faceDetectionAsyncTask.setCascadeFaceDetector(cascadeFaceDetector);
				faceDetectionAsyncTask.execute();
			} else if (faceDetectSyncTaskStatus.equals(AsyncTask.Status.RUNNING)) {
				Log.i(TAG, "faceDetectAsyncTask is running");
			}
			if (faceDetectionAsyncTask != null && foundFaceFromAsyncTask == null) {
				foundFaceFromAsyncTask = faceDetectionAsyncTask.getFoundFace();
			}
			DrawingUtils.drawRect(foundFaceFromAsyncTask, currentlyUsedFrame, DrawingConstants.FACE_RECT_COLOR);
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_JAVA:
			currentlyUsedFrame = colorSegmentationFaceDetector.detectFaceYCrCb(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_CPP:
			SegmentSkin(mRgba.getNativeObjAddr(), currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_JAVA:
			Rect foundFace;
			if (isFaceTracking) {
				foundFace = cascadeFaceDetector.findFace(mGray);
			} else {
				foundFace = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFace != null) {
				DrawingUtils.drawRect(foundFace, currentlyUsedFrame, DrawingConstants.FACE_RECT_COLOR);
				Rect[] eyes = cascadeEyesDetector.findEyes(mGray, foundFace, false);
				DrawingUtils.drawRects(eyes, currentlyUsedFrame, DrawingConstants.EYES_RECT_COLOR);
			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_CPP:
			FindEyes(currentlyUsedFrame.getNativeObjAddr(), currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_FIND_MOUTH_CASCADE_JAVA:
			Rect foundFace2;
			if (isFaceTracking) {
				foundFace2 = cascadeFaceDetector.findFace(currentlyUsedFrame);
			} else {
				foundFace2 = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFace2 != null) {
				DrawingUtils.drawRect(foundFace2, currentlyUsedFrame, DrawingConstants.FACE_RECT_COLOR);
				Rect mouth = cascadeMouthDetector.findMouth(currentlyUsedFrame, foundFace2.clone());
				DrawingUtils.drawRect(mouth, currentlyUsedFrame, DrawingConstants.MOUTH_RECT_COLOR);
				if (isMouthWithEyes) {
					Rect[] eyes = cascadeEyesDetector.findEyes(currentlyUsedFrame, foundFace2.clone(), false);
					DrawingUtils.drawRects(eyes, currentlyUsedFrame, DrawingConstants.EYES_RECT_COLOR);
					break;
				}

			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_NOSE_CASCADE_JAVA:
			Rect foundFace3;
			if (isFaceTracking) {
				foundFace3 = cascadeFaceDetector.findFace(currentlyUsedFrame);
			} else {
				foundFace3 = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFace3 != null) {
				DrawingUtils.drawRect(foundFace3, currentlyUsedFrame, DrawingConstants.FACE_RECT_COLOR);
				Rect[] noses = cascadeNoseDetector.findNoses(currentlyUsedFrame, foundFace3);
				DrawingUtils.drawRects(noses, currentlyUsedFrame, DrawingConstants.NOSE_RECT_COLOR);
			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_JAVA:
			MatOfPoint cornersPoints = new MatOfPoint();
			Imgproc.goodFeaturesToTrack(mGray, cornersPoints, 100, 0.01, 10);
			int rows = cornersPoints.rows();
			List<Point> listOfPoints = cornersPoints.toList();
			for (int x = 0; x < rows; x++) {
				Core.circle(currentlyUsedFrame, listOfPoints.get(x), 5, DrawingConstants.RED);
			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_CPP:
			FindFeatures(mGray.getNativeObjAddr(), currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_FIND_CORNER_HARRIS_JAVA:
			Mat dst = Mat.zeros(mGray.size(), CvType.CV_32FC1);
			Mat dst_norm = new Mat();
			Mat dst_norm_scaled = new Mat();

			int blockSize = 2;
			int apertureSize = 3;
			double k = 0.04;

			Imgproc.cornerHarris(mGray, dst, blockSize, apertureSize, k, Imgproc.BORDER_DEFAULT);
			Core.normalize(dst, dst_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());
			Core.convertScaleAbs(dst_norm, dst_norm_scaled);

			currentlyUsedFrame = dst_norm_scaled;
			break;

		case ViewModesConstants.VIEW_MODE_FIND_CORNER_HARRIS_CPP:
			FindCornerHarris(mGray.getNativeObjAddr(), currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_OPTICAL_FLOW_JAVA:
			if (mMOP2fptsPrev.rows() == 0) {
				Imgproc.cvtColor(mRgba, matOpFlowThis, Imgproc.COLOR_RGBA2GRAY);
				matOpFlowThis.copyTo(matOpFlowPrev);
				Imgproc.goodFeaturesToTrack(matOpFlowPrev, MOPcorners, iGFFTMax, 0.05, 20);
				mMOP2fptsPrev.fromArray(MOPcorners.toArray());
				mMOP2fptsPrev.copyTo(mMOP2fptsSafe);
			} else {
				matOpFlowThis.copyTo(matOpFlowPrev);
				Imgproc.cvtColor(mRgba, matOpFlowThis, Imgproc.COLOR_RGBA2GRAY);
				Imgproc.goodFeaturesToTrack(matOpFlowThis, MOPcorners, iGFFTMax, 0.05, 20);
				mMOP2fptsThis.fromArray(MOPcorners.toArray());
				mMOP2fptsSafe.copyTo(mMOP2fptsPrev);
				mMOP2fptsThis.copyTo(mMOP2fptsSafe);
			}

			Video.calcOpticalFlowPyrLK(matOpFlowPrev, matOpFlowThis, mMOP2fptsPrev, mMOP2fptsThis, mMOBStatus, mMOFerr);

			cornersPrev = mMOP2fptsPrev.toList();
			cornersThis = mMOP2fptsThis.toList();
			byteStatus = mMOBStatus.toList();

			int y = byteStatus.size() - 1;

			for (int x = 0; x < y; x++) {
				if (byteStatus.get(x) == 1) {
					Point pt = cornersThis.get(x);
					Point pt2 = cornersPrev.get(x);
					Core.circle(currentlyUsedFrame, pt, 5, DrawingConstants.RED, 5);
					Core.line(currentlyUsedFrame, pt, pt2, DrawingConstants.RED, 3);
				}
			}

			break;

		case ViewModesConstants.VIEW_MODE_OPTICAL_FLOW_CPP:
			CalculateOpticalFlow(mRgba.getNativeObjAddr(), currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_STASM:

			int[] points = FindFaceLandmarks(mGray.getNativeObjAddr());

			if (points[0] > 0) {
				Point pt;
				for (int i = 0; i < points.length / 2; ++i) {
					pt = new Point(points[i * 2], points[i * 2 + 1]);
					Core.circle(currentlyUsedFrame, pt, 4, DrawingConstants.GREEN);
				}
			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_ALL:
			Rect foundFaceInDetection = new Rect(0, 0, mGray.width(), mGray.height());
			Rect boundingBox = new Rect(0, 0, mGray.width(), mGray.height());
			double boundingMultiplier = 0.1;
			boundingBox.x += boundingMultiplier * mGray.width();
			boundingBox.width -= 2 * boundingMultiplier * mGray.width();
			foundFaceInDetection = cascadeFaceDetector.findFace(mGray, boundingBox);
			if (foundFaceInDetection == null) {
				foundFaceInDetection = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFaceInDetection != null) {
				Rect[] eyes = null;
				Rect mouth = null;
				Rect nose = null;

				Rect foundFaceForEyes = foundFaceInDetection.clone();
				eyes = cascadeEyesDetector.findEyes(mGray, foundFaceForEyes, false);
				if (eyes == null || eyes.length == 0) {
					eyes = cascadeEyesDetector.getLastFoundEyes();
				}

				Rect foundFaceForMouth = foundFaceInDetection.clone();
				mouth = cascadeMouthDetector.findMouth(mGray, foundFaceForMouth);
				if (mouth == null) {
					mouth = cascadeMouthDetector.getLastFoundMouth();
				}

				Rect foundFaceForNose = foundFaceInDetection.clone();
				nose = cascadeNoseDetector.findNose(mGray, foundFaceForNose);
				if (nose == null) {
					nose = cascadeNoseDetector.getLastFoundNose();
				}

				ArrayList<Float> characteristicPoints = new ArrayList<Float>();
				if (eyes != null && eyes.length > 0) {
					characteristicPoints.add((float) VisualUtils.getCentrePoint(eyes[0]).x);
					characteristicPoints.add((float) VisualUtils.getCentrePoint(eyes[0]).y);
					if (eyes.length == 2) {
						characteristicPoints.add((float) VisualUtils.getCentrePoint(eyes[1]).x);
						characteristicPoints.add((float) VisualUtils.getCentrePoint(eyes[1]).y);
					}
				}
				if (nose != null) {
					characteristicPoints.add((float) VisualUtils.getCentrePoint(nose).x);
					characteristicPoints.add((float) VisualUtils.getCentrePoint(nose).y);
				}
				if (mouth != null) {
					characteristicPoints.add((float) VisualUtils.getCentrePoint(mouth).x - 5);
					characteristicPoints.add((float) VisualUtils.getCentrePoint(mouth).y);
					characteristicPoints.add((float) VisualUtils.getCentrePoint(mouth).x + 5);
					characteristicPoints.add((float) VisualUtils.getCentrePoint(mouth).y);
				}
				float[] floatArray = new float[characteristicPoints.size()];
				int iter = 0;
				for (Float f : characteristicPoints) {
					floatArray[iter++] = (f != null ? f : Float.NaN);
				}
				if (floatArray != null && floatArray.length == 10) {
					int[] kpoints = AddFaceLandmarks(mGray.getNativeObjAddr(), floatArray);
					if (kpoints[0] > 0) {
						Point pt;
						for (int i = 0; i < kpoints.length / 2; ++i) {
							pt = new Point(kpoints[i * 2], kpoints[i * 2 + 1]);
							Core.circle(mRgba, pt, 4, DrawingConstants.GREEN);
						}
					}
				}

				DrawingUtils.drawRect(foundFaceInDetection, mRgba, DrawingConstants.FACE_RECT_COLOR);
				DrawingUtils.drawRects(eyes, mRgba, DrawingConstants.EYES_RECT_COLOR);
				DrawingUtils.drawRect(mouth, mRgba, DrawingConstants.MOUTH_RECT_COLOR);
				DrawingUtils.drawRect(nose, mRgba, DrawingConstants.NOSE_RECT_COLOR);
			}

			break;

		case ViewModesConstants.VIEW_MODE_CANNY_EDGE:
			edgeDetectionAlgorithm.cannyEdgeDetection(currentlyUsedFrame);
			break;
			
		case ViewModesConstants.VIEW_MODE_SOBEL_EDGE:
			edgeDetectionAlgorithm.sobelEdgeDetection(currentlyUsedFrame);
			break;
			
		case ViewModesConstants.VIEW_MODE_LAPLACIAN_EDGE:
			edgeDetectionAlgorithm.laplacianEdgeDetection(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_SOBEL_EDGE_ADVANCED:
			edgeDetectionAlgorithm.sobelAdvancedEdgeDetection(currentlyUsedFrame);
			break;
			
		case ViewModesConstants.VIEW_MODE_LAPLACIAN_EDGE_ADVANCED:
			edgeDetectionAlgorithm.laplacianAdvancedEdgeDetection(currentlyUsedFrame);
			break;
			
		case ViewModesConstants.VIEW_MODE_BIN_STANDARD:
			binarizationAlgorithm.standardBinarization(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_BIN_STANDARD_TRUNC:
			binarizationAlgorithm.standardTruncBinarization(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_BIN_OTSU:
			binarizationAlgorithm.otsuBinarization(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_BIN_ADAPTIVE_MEAN:
			binarizationAlgorithm.adaptiveMeanBinarization(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_BIN_ADAPTIVE_GAUSSIAN:
			binarizationAlgorithm.adaptiveGaussianBinarization(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_START_DROWSINESS_DETECTION:
			currentlyUsedFrame = drowsinessDetector.processDetection(mGray, mRgba);
			break;

		}

		return currentlyUsedFrame;
	}

	public boolean onOptionsItemSelected(MenuItem item) {
		if (item.getGroupId() == 4) {
			mEqHistMode = MainActivityHelper.getHistMode(item);
		} else if (item.getGroupId() == 3) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_MOUTH_CASCADE_JAVA;
				isMouthWithEyes = false;
				break;
			case 1:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_MOUTH_CASCADE_JAVA;
				isMouthWithEyes = true;
				break;
			}
		} else {
			mViewMode = MainActivityHelper.getViewMode(item);
		}
		return true;
	}

	public void showOptionsMenu(View v) {
		if (mCustomCameraView != null) {
			Log.i(TAG, ((CustomCameraView) mCustomCameraView).getCameraInfo());
		}
		openOptionsMenu();
	}

	public void showCascadeFaceFilesMenu(final View v) {
		PopupMenu popup = new PopupMenu(this, v);
		MenuInflater inflater = popup.getMenuInflater();
		inflater.inflate(R.menu.cascade_face_files_menu, popup.getMenu());
		popup.setOnMenuItemClickListener(new OnMenuItemClickListener() {
			@Override
			public boolean onMenuItemClick(MenuItem menuitem) {
				int faceResourceId = getResources().getIdentifier(menuitem.getTitle().toString(), "raw", getPackageName());
				InputStream is = getResources().openRawResource(faceResourceId);
				File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
				cascadeFaceDetector.setCascadeFileName(menuitem.getTitle().toString().concat(".xml"));
				File faceFile = cascadeFaceDetector.prepare(is, cascadeDir);
				PrepareFindFace(faceFile.getAbsolutePath());
				return false;
			}
		});
		popup.show();
	}

	public void changeIsFaceTracking(final View v) {
		if (isFaceTracking) {
			isFaceTracking = false;
		} else {
			isFaceTracking = true;
		}
		Toast.makeText(this, "Face tracking: " + String.valueOf(isFaceTracking), Toast.LENGTH_SHORT).show();
	}

	public void showCascadeEyesFilesMenu(final View v) {
		PopupMenu popup = new PopupMenu(this, v);
		MenuInflater inflater = popup.getMenuInflater();
		inflater.inflate(R.menu.cascade_eyes_files_menu, popup.getMenu());
		popup.setOnMenuItemClickListener(new OnMenuItemClickListener() {
			@Override
			public boolean onMenuItemClick(MenuItem menuitem) {
				int eyesResourceId = getResources().getIdentifier(menuitem.getTitle().toString(), "raw", getPackageName());
				InputStream is = getResources().openRawResource(eyesResourceId);
				File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
				cascadeEyesDetector.setCascadeFileName(menuitem.getTitle().toString().concat(".xml"));
				File faceFile = cascadeEyesDetector.prepare(is, cascadeDir);
				if (faceFile.getName().contains("eyepair")) {
					cascadeEyesDetector.setBothEyes(true);
				} else {
					cascadeEyesDetector.setBothEyes(false);
				}
				return false;
			}
		});
		popup.show();
	}

	public void changeResolution(final View v) {
		if (mCustomCameraView instanceof CustomCameraView) {
			final CustomCameraView mCustomCameraView = (CustomCameraView) this.mCustomCameraView;
			PopupMenu popup = new PopupMenu(this, v);
			mResolutionList = mCustomCameraView.getResolutionList();
			mResolutionMenuItems = new MenuItem[mResolutionList.size()];

			ListIterator<android.hardware.Camera.Size> resolutionItr = mResolutionList.listIterator();
			int idx = 0;
			while (resolutionItr.hasNext()) {
				android.hardware.Camera.Size element = resolutionItr.next();
				mResolutionMenuItems[idx] = popup.getMenu().add(2, idx, Menu.NONE,
						Integer.valueOf(element.width).toString() + "x" + Integer.valueOf(element.height).toString());
				idx++;
			}

			popup.setOnMenuItemClickListener(new OnMenuItemClickListener() {

				@Override
				public boolean onMenuItemClick(MenuItem item) {
					int id = item.getItemId();
					android.hardware.Camera.Size resolution = mResolutionList.get(id);
					mCustomCameraView.setResolution(resolution);
					resolution = mCustomCameraView.getResolution();
					String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
					Toast.makeText(getBaseContext(), caption, Toast.LENGTH_SHORT).show();
					return false;
				}
			});
			popup.show();
		} else {
			Toast.makeText(getBaseContext(), "This camera view does not support changing the resolution", Toast.LENGTH_SHORT).show();
		}
	}

	public void changeImageColorSpace(View v) {
		if (isCurrentFrameRgb) {
			Toast.makeText(getBaseContext(), "Gray", Toast.LENGTH_SHORT).show();
			isCurrentFrameRgb = false;
		} else {
			Toast.makeText(getBaseContext(), "RGB", Toast.LENGTH_SHORT).show();
			isCurrentFrameRgb = true;
		}
	}

	public void changeDetectionFlag(View v) {
		PopupMenu popup = new PopupMenu(this, v);
		MenuInflater inflater = popup.getMenuInflater();
		inflater.inflate(R.menu.cascade_detection_flag_menu, popup.getMenu());
		popup.setOnMenuItemClickListener(new OnMenuItemClickListener() {
			@Override
			public boolean onMenuItemClick(MenuItem menuitem) {
				int detectionFlag = Objdetect.CASCADE_SCALE_IMAGE;
				String menuItemTitle = menuitem.getTitle().toString();
				if (menuItemTitle.equals("CV_HAAR_DO_CANNY_PRUNING")) {
					detectionFlag = Objdetect.CASCADE_DO_CANNY_PRUNING;
				} else if (menuItemTitle.equals("CV_HAAR_SCALE_IMAGE")) {
					detectionFlag = Objdetect.CASCADE_SCALE_IMAGE;
				} else if (menuItemTitle.equals("CV_HAAR_FIND_BIGGEST_OBJECT")) {
					detectionFlag = Objdetect.CASCADE_FIND_BIGGEST_OBJECT;
				} else if (menuItemTitle.equals("CV_HAAR_DO_ROUGH_SEARCH")) {
					detectionFlag = Objdetect.CASCADE_DO_ROUGH_SEARCH;
				}
				cascadeFaceDetector.setDetectionFlag(detectionFlag);
				cascadeEyesDetector.setDetectionFlag(detectionFlag);
				cascadeMouthDetector.setDetectionFlag(detectionFlag);
				cascadeNoseDetector.setDetectionFlag(detectionFlag);
				return false;
			}
		});
		popup.show();
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		Log.i(TAG, "called onCreateOptionsMenu");
		SubMenu mItemPreviewFindFaceMenu = menu.addSubMenu("Find face");
		SubMenu mItemPreviewFindEyesMenu = menu.addSubMenu("Find eyes");
		SubMenu mItemPreviewFindMouthMenu = menu.addSubMenu("Find mouth");
		menu.add(6, 0, Menu.NONE, "Find nose");
		menu.add(7, 0, Menu.NONE, "Find all");
		SubMenu mItemPreviewEqHistMenu = menu.addSubMenu("Equalize histogram");
		SubMenu mItemPreviewBinarizationMenu = menu.addSubMenu("Binarization");
		SubMenu mItemPreviewEdgeDetectionMenu = menu.addSubMenu("Edge detection");
		SubMenu mItemPreviewOtherMenu = menu.addSubMenu("Other");

		mItemPreviewFindFaceMenu.add(1, 0, Menu.NONE, "Snapdragon");
		mItemPreviewFindFaceMenu.add(1, 1, Menu.NONE, "Haar (Java)");
		mItemPreviewFindFaceMenu.add(1, 2, Menu.NONE, "Haar (Cpp)");
		mItemPreviewFindFaceMenu.add(1, 3, Menu.NONE, "Haar (Java AsyncTask)");
		mItemPreviewFindFaceMenu.add(1, 4, Menu.NONE, "Color segmentation (Java)");
		mItemPreviewFindFaceMenu.add(1, 5, Menu.NONE, "Color segmentation (Cpp)");

		mItemPreviewFindEyesMenu.add(2, 0, Menu.NONE, "Java");
		mItemPreviewFindEyesMenu.add(2, 1, Menu.NONE, "Cpp");

		mItemPreviewFindMouthMenu.add(3, 0, Menu.NONE, "Without eyes");
		mItemPreviewFindMouthMenu.add(3, 1, Menu.NONE, "With eyes");

		mItemPreviewEqHistMenu.add(4, 0, Menu.NONE, "None");
		mItemPreviewEqHistMenu.add(4, 1, Menu.NONE, "Standard (Java)");
		mItemPreviewEqHistMenu.add(4, 2, Menu.NONE, "Standard (Cpp)");
		mItemPreviewEqHistMenu.add(4, 3, Menu.NONE, "CLAHE");
		
		mItemPreviewEdgeDetectionMenu.add(9, 0, Menu.NONE, "Canny");
		mItemPreviewEdgeDetectionMenu.add(9, 1, Menu.NONE, "Sobel");
		mItemPreviewEdgeDetectionMenu.add(9, 2, Menu.NONE, "Laplacian");
		mItemPreviewEdgeDetectionMenu.add(9, 3, Menu.NONE, "Sobel (advanced)");
		mItemPreviewEdgeDetectionMenu.add(9, 4, Menu.NONE, "Laplacian (advanced)");
		
		mItemPreviewOtherMenu.add(5, 0, Menu.NONE, "Find features (Java)");
		mItemPreviewOtherMenu.add(5, 1, Menu.NONE, "Find features (Cpp)");
		mItemPreviewOtherMenu.add(5, 2, Menu.NONE, "Find corner Harris (Java)");
		mItemPreviewOtherMenu.add(5, 3, Menu.NONE, "Find corner Harris (Cpp)");
		mItemPreviewOtherMenu.add(5, 4, Menu.NONE, "Optical flow (Java)");
		mItemPreviewOtherMenu.add(5, 5, Menu.NONE, "Optical flow (Cpp)");
		mItemPreviewOtherMenu.add(5, 6, Menu.NONE, "STASM");

		mItemPreviewBinarizationMenu.add(8, 0, Menu.NONE, "Standard");
		mItemPreviewBinarizationMenu.add(8, 1, Menu.NONE, "Standard (trunc)");
		mItemPreviewBinarizationMenu.add(8, 2, Menu.NONE, "Otsu");
		mItemPreviewBinarizationMenu.add(8, 3, Menu.NONE, "Adaptive mean");
		mItemPreviewBinarizationMenu.add(8, 4, Menu.NONE, "Adaptive gaussian");

		return true;
	}

	public void resetView(View v) {
		mViewMode = ViewModesConstants.VIEW_MODE_NONE;
		mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_NONE;
	}

	public void startDrowsinessDetection(View v) {
		AlertDialog dialog;
		AlertDialog.Builder builder = new AlertDialog.Builder(this);
		builder.setTitle("Select algorithm elements");
		CharSequence[] items = drowsinessDetector.getItems();
		boolean[] checkedItems = drowsinessDetector.getDetectionFlags();
		builder.setMultiChoiceItems(items, checkedItems, new Dialog.OnMultiChoiceClickListener() {
			@Override
			public void onClick(DialogInterface dialog, int indexSelected, boolean isChecked) {
				drowsinessDetector.setDetectionElementsById(indexSelected, isChecked);
			}
		});
		builder.setPositiveButton("Ok", new OnClickListener() {
			@Override
			public void onClick(DialogInterface dialog, int which) {
				AlertDialog dialog1;
				AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
				builder.setTitle("Select eyes detection algorithm");
				CharSequence[] algorithms = new CharSequence[] { "Pair of eyes", "Separate eyes" };
				drowsinessDetector.setSeparateEyesDetection(false);
				builder.setSingleChoiceItems(algorithms, 0, new OnClickListener() {
					@Override
					public void onClick(DialogInterface arg0, int arg1) {
						if (arg1 == 0) {
							drowsinessDetector.setSeparateEyesDetection(false);
						} else {
							drowsinessDetector.setSeparateEyesDetection(true);
						}
					}
				});
				builder.setPositiveButton("Ok", new OnClickListener() {
					@Override
					public void onClick(DialogInterface arg0, int arg1) {
						AlertDialog dialog2;
						AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
						builder.setTitle("Select eye openess algorithm");
						CharSequence[] algorithms = new CharSequence[] { "DarkBright - adaptive binarization","DarkBright - simple binarization", "Edge detection - Canny", "Edge detection - Laplacian", "Do nothing" };
						drowsinessDetector.setCannyAlgorithmUsed(false);
						drowsinessDetector.setDoNothing(false);
						drowsinessDetector.setSimpleBinarizationUsed(false);
						drowsinessDetector.setSobelAlgorithmUsed(false);
						builder.setSingleChoiceItems(algorithms, 0, new OnClickListener() {
							@Override
							public void onClick(DialogInterface arg0, int arg1) {
								if (arg1 == 0) {
									drowsinessDetector.setCannyAlgorithmUsed(false);
								} else if (arg1 == 1){
									drowsinessDetector.setSimpleBinarizationUsed(true);
								} else  if (arg1 == 2){
									drowsinessDetector.setCannyAlgorithmUsed(true);
								} else if (arg1 == 3){
									drowsinessDetector.setSobelAlgorithmUsed(true);
								} else if (arg1 == 4){
									drowsinessDetector.setDoNothing(true);
								}
							}
						});
						builder.setPositiveButton("Ok", new OnClickListener() {
							@Override
							public void onClick(DialogInterface arg0, int arg1) {
								mViewMode = ViewModesConstants.VIEW_MODE_START_DROWSINESS_DETECTION;
							}
						});
						dialog2 = builder.create();
						dialog2.show();
					}
				});
				dialog1 = builder.create();
				dialog1.show();
			}
		});
		dialog = builder.create();
		dialog.show();
	}

	public void showHideGui(View v) {
		RelativeLayout mainLayout = (RelativeLayout) findViewById(R.id.mainLayout);
		if (isGuiHidden) {
			for (int i = 0; i < mainLayout.getChildCount(); i++) {
				View insideView = mainLayout.getChildAt(i);
				changeVisibiltyOfChildViews(insideView, View.VISIBLE);
			}
			isGuiHidden = false;
		} else {
			for (int i = 0; i < mainLayout.getChildCount(); i++) {
				View insideView = mainLayout.getChildAt(i);
				changeVisibiltyOfChildViews(insideView, View.INVISIBLE);
			}
			isGuiHidden = true;
		}
	}

	private void changeVisibiltyOfChildViews(View parentView, int visibiltyFlag) {
		if (parentView instanceof ViewGroup) {
			for (int i = 0; i < ((ViewGroup) parentView).getChildCount(); i++) {
				changeVisibiltyOfChildViews(((ViewGroup) parentView).getChildAt(i), visibiltyFlag);
			}
		} else {
			if (!(parentView instanceof CustomCameraView || parentView.getId() == R.id.showHideGui)) {
				parentView.setVisibility(visibiltyFlag);
			}
		}
	}

	public void chooseObjectToChangeItsSize(final View v) {
		PopupMenu popup = new PopupMenu(this, v);
		MenuInflater inflater = popup.getMenuInflater();
		inflater.inflate(R.menu.detection_things_menu, popup.getMenu());
		popup.setOnMenuItemClickListener(new OnMenuItemClickListener() {
			@Override
			public boolean onMenuItemClick(MenuItem menuitem) {
				String menuItemTitle = menuitem.getTitle().toString();
				if (menuItemTitle.equals("Mouth")) {
					isMouthChosen = true;
					isNoseChosen = false;
					minObjectValueText.setText(String.valueOf(cascadeMouthDetector.getmRelativeMinObjectSize()));
					minObjectSeekBar.setProgress((int) (cascadeMouthDetector.getmRelativeMinObjectSize() * 100));
					maxObjectValueText.setText(String.valueOf(cascadeMouthDetector.getmRelativeMaxObjectSize()));
					maxObjectSeekBar.setProgress((int) (cascadeMouthDetector.getmRelativeMaxObjectSize() * 100));
					Button thisButton = (Button) v;
					thisButton.setText("Size: mouth");
				} else if (menuItemTitle.equals("Nose")) {
					isNoseChosen = true;
					isMouthChosen = false;
					minObjectValueText.setText(String.valueOf(cascadeNoseDetector.getmRelativeMinObjectSize()));
					minObjectSeekBar.setProgress((int) (cascadeNoseDetector.getmRelativeMinObjectSize() * 100));
					maxObjectValueText.setText(String.valueOf(cascadeNoseDetector.getmRelativeMaxObjectSize()));
					maxObjectSeekBar.setProgress((int) (cascadeNoseDetector.getmRelativeMaxObjectSize() * 100));
					Button thisButton = (Button) v;
					thisButton.setText("Size: nose");
				}
				return false;
			}
		});
		popup.show();
	}

	public void showSeekBars(View v) {
		AlertDialog dialog;
		AlertDialog.Builder builder = new AlertDialog.Builder(this);
		builder.setTitle("Select visible seek bars");
		ArrayList<String> seekBarsStrings = new ArrayList<String>();
		seekBarsStrings.addAll(verticalSeekBars.keySet());
		final CharSequence[] items = seekBarsStrings.toArray(new CharSequence[seekBarsStrings.size()]);
		final boolean[] checkedItems = { true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false};
		final Map<Integer, Integer> samePlaceVerticalSeekBars = initSamePlaceVerticalSeekBar();
		builder.setMultiChoiceItems(items, checkedItems, new Dialog.OnMultiChoiceClickListener() {
			@Override
			public void onClick(DialogInterface insideDialog, int indexSelected, boolean isChecked) {
				CharSequence itemTitle = items[indexSelected];
				if (isChecked) {
					verticalSeekBars.get(itemTitle).setVisibility(View.VISIBLE);
					verticalSeekBarsTextNames.get(itemTitle).setVisibility(View.VISIBLE);
					verticalSeekBarsTextValues.get(itemTitle).setVisibility(View.VISIBLE);
					int noOfItemToHide = samePlaceVerticalSeekBars.get(indexSelected);
					CharSequence itemToHide = items[noOfItemToHide];
					verticalSeekBars.get(itemToHide).setVisibility(View.INVISIBLE);
					verticalSeekBarsTextNames.get(itemToHide).setVisibility(View.INVISIBLE);
					verticalSeekBarsTextValues.get(itemToHide).setVisibility(View.INVISIBLE);
					checkedItems[noOfItemToHide] = false;
				} else {
					verticalSeekBars.get(itemTitle).setVisibility(View.INVISIBLE);
					verticalSeekBarsTextNames.get(itemTitle).setVisibility(View.INVISIBLE);
					verticalSeekBarsTextValues.get(itemTitle).setVisibility(View.INVISIBLE);
					int noOfItemToHide = samePlaceVerticalSeekBars.get(indexSelected);
					CharSequence itemToHide = items[noOfItemToHide];
					verticalSeekBars.get(itemToHide).setVisibility(View.VISIBLE);
					verticalSeekBarsTextNames.get(itemToHide).setVisibility(View.VISIBLE);
					verticalSeekBarsTextValues.get(itemToHide).setVisibility(View.VISIBLE);
					checkedItems[noOfItemToHide] = true;
				}

			}
		});
		dialog = builder.create();
		dialog.show();
	}

	public HashMap<Integer, Integer> initSamePlaceVerticalSeekBar() {
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		map.put(0, 14);
		map.put(14, 0);
		map.put(1, 15);
		map.put(15, 1);
		map.put(2, 11);
		map.put(11, 2);
		map.put(3, 12);
		map.put(12, 3);
		map.put(4, 17);
		map.put(17, 4);
		map.put(5, 13);
		map.put(13, 5);
		map.put(8, 16);
		map.put(16, 8);
		return map;
	}

	public void initVerticalSeekBars() {

		scaleFactorSeekBar = (VerticalSeekBar) findViewById(R.id.scaleFactorSeekBar);
		verticalSeekBars.put("scaleFactor", scaleFactorSeekBar);
		scaleFactorValueText = (TextView) findViewById(R.id.scaleFactorValueText);
		verticalSeekBarsTextValues.put("scaleFactor", scaleFactorValueText);
		verticalSeekBarsTextNames.put("scaleFactor", (TextView) findViewById(R.id.scaleFactorNameText));
		scaleFactorValueText.setText(String.valueOf(cascadeFaceDetector.getScaleFactor()));
		scaleFactorSeekBar.setProgress((int) ((cascadeFaceDetector.getScaleFactor() / DetectorConstants.MAX_SCALE_FACTOR) * 100));
		scaleFactorSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float currentValue = ((progress * (DetectorConstants.MAX_SCALE_FACTOR - DetectorConstants.MIN_SCALE_FACTOR)) / 100)
						+ DetectorConstants.MIN_SCALE_FACTOR;
				scaleFactorValueText.setText(String.valueOf(MathUtils.round(currentValue, 2)));
				switch (mViewMode) {
				case ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_JAVA:
					cascadeFaceDetector.setScaleFactor(currentValue);
					break;

				case ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_JAVA:
					cascadeEyesDetector.setScaleFactor(currentValue);
					break;
				}
			}
		});

		minNeighsSeekBar = (VerticalSeekBar) findViewById(R.id.minNeighsSeekBar);
		verticalSeekBars.put("minNeighs", minNeighsSeekBar);
		minNeighsValueText = (TextView) findViewById(R.id.minNeighboursValueText);
		verticalSeekBarsTextValues.put("minNeighs", minNeighsValueText);
		verticalSeekBarsTextNames.put("minNeighs", (TextView) findViewById(R.id.minNeigbrsNameText));
		minNeighsValueText.setText(String.valueOf(cascadeFaceDetector.getMinNeighbours()));
		minNeighsSeekBar.setProgress(cascadeFaceDetector.getMinNeighbours() * 100 / DetectorConstants.MAX_MIN_NEIGHBOURS);
		minNeighsSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_MIN_NEIGHBOURS - DetectorConstants.MIN_MIN_NEIGHBOURS)) / 100)
						+ DetectorConstants.MIN_MIN_NEIGHBOURS;
				minNeighsValueText.setText(String.valueOf(MathUtils.round(currentValue, 2)));
				switch (mViewMode) {
				case ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_JAVA:
					cascadeFaceDetector.setMinNeighbours(currentValue);
					break;

				case ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_JAVA:
					cascadeEyesDetector.setMinNeighbours(currentValue);
					break;
				}
			}
		});

		minFaceSeekBar = (VerticalSeekBar) findViewById(R.id.minFaceSeekBar);
		verticalSeekBars.put("minFace", minFaceSeekBar);
		minFaceValueText = (TextView) findViewById(R.id.minFaceValueText);
		verticalSeekBarsTextValues.put("minFace", minFaceValueText);
		verticalSeekBarsTextNames.put("minFace", (TextView) findViewById(R.id.minFaceNameText));
		minFaceValueText.setText(String.valueOf(cascadeFaceDetector.getmRelativeMinObjectSize()));
		minFaceSeekBar.setProgress((int) (cascadeFaceDetector.getmRelativeMinObjectSize() * 100));
		minFaceSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float currentValue = (float) ((float) progress / 100.0);
				float maxFace = Float.parseFloat(maxFaceValueText.getText().toString());
				if (currentValue > maxFace) {
					currentValue = maxFace;
				}
				minFaceValueText.setText(String.valueOf(currentValue));
				cascadeFaceDetector.setSizeManuallyChanged(true);
				cascadeFaceDetector.setmRelativeMinObjectSize(currentValue);
			}
		});

		maxFaceSeekBar = (VerticalSeekBar) findViewById(R.id.maxFaceSeekBar);
		verticalSeekBars.put("maxFace", maxFaceSeekBar);
		maxFaceValueText = (TextView) findViewById(R.id.maxFaceValueText);
		verticalSeekBarsTextValues.put("maxFace", maxFaceValueText);
		verticalSeekBarsTextNames.put("maxFace", (TextView) findViewById(R.id.maxFaceNameText));
		maxFaceValueText.setText(String.valueOf(cascadeFaceDetector.getmRelativeMaxObjectSize()));
		maxFaceSeekBar.setProgress((int) (cascadeFaceDetector.getmRelativeMaxObjectSize() * 100));
		maxFaceSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float currentValue = (float) ((float) progress / 100.0);
				float minFace = Float.parseFloat(minFaceValueText.getText().toString());
				if (currentValue < minFace) {
					currentValue = minFace;
				}
				maxFaceValueText.setText(String.valueOf(currentValue));
				cascadeFaceDetector.setSizeManuallyChanged(true);
				cascadeFaceDetector.setmRelativeMaxObjectSize(currentValue);
			}
		});

		minEyeSeekBar = (VerticalSeekBar) findViewById(R.id.minEyeSeekBar);
		verticalSeekBars.put("minEye", minEyeSeekBar);
		minEyeValueText = (TextView) findViewById(R.id.minEyeValueText);
		verticalSeekBarsTextValues.put("minEye", minEyeValueText);
		verticalSeekBarsTextNames.put("minEye", (TextView) findViewById(R.id.minEyeNameText));
		minEyeValueText.setText(String.valueOf(cascadeEyesDetector.getmRelativeMinObjectSize()));
		minEyeSeekBar.setProgress((int) (cascadeEyesDetector.getmRelativeMinObjectSize() * 100));
		minEyeSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float currentValue = (float) ((float) progress / 100.0);
				float maxEye = Float.parseFloat(maxEyeValueText.getText().toString());
				if (currentValue > maxEye) {
					currentValue = maxEye;
				}
				minEyeValueText.setText(String.valueOf(currentValue));
				cascadeEyesDetector.setSizeManuallyChanged(true);
				cascadeEyesDetector.setmRelativeMinObjectSize(currentValue);
			}
		});

		maxEyeSeekBar = (VerticalSeekBar) findViewById(R.id.maxEyeSeekBar);
		verticalSeekBars.put("maxEye", maxEyeSeekBar);
		maxEyeValueText = (TextView) findViewById(R.id.maxEyeValueText);
		verticalSeekBarsTextValues.put("maxEye", maxEyeValueText);
		verticalSeekBarsTextNames.put("maxEye", (TextView) findViewById(R.id.maxEyeNameText));
		maxEyeValueText.setText(String.valueOf(cascadeEyesDetector.getmRelativeMaxObjectSize()));
		maxEyeSeekBar.setProgress((int) (cascadeEyesDetector.getmRelativeMaxObjectSize() * 100));
		maxEyeSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float currentValue = (float) ((float) progress / 100.0);
				float minEye = Float.parseFloat(minEyeValueText.getText().toString());
				if (currentValue < minEye) {
					currentValue = minEye;
				}
				maxEyeValueText.setText(String.valueOf(currentValue));
				cascadeEyesDetector.setSizeManuallyChanged(true);
				cascadeEyesDetector.setmRelativeMaxObjectSize(currentValue);
			}
		});

		minObjectSeekBar = (VerticalSeekBar) findViewById(R.id.minObjectSeekBar);
		verticalSeekBars.put("minObject", minObjectSeekBar);
		minObjectValueText = (TextView) findViewById(R.id.minObjectValueText);
		verticalSeekBarsTextValues.put("minObject", minObjectValueText);
		verticalSeekBarsTextNames.put("minObject", (TextView) findViewById(R.id.minObjectNameText));
		if (isMouthChosen) {
			minObjectValueText.setText(String.valueOf(cascadeMouthDetector.getmRelativeMinObjectSize()));
			minObjectSeekBar.setProgress((int) (cascadeMouthDetector.getmRelativeMinObjectSize() * 100));
		} else if (isNoseChosen) {
			minObjectValueText.setText(String.valueOf(cascadeNoseDetector.getmRelativeMinObjectSize()));
			minObjectSeekBar.setProgress((int) (cascadeNoseDetector.getmRelativeMinObjectSize() * 100));
		}
		minObjectSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float currentValue = (float) ((float) progress / 100.0);
				float maxObject = Float.parseFloat(maxObjectValueText.getText().toString());
				if (currentValue > maxObject) {
					currentValue = maxObject;
				}
				minObjectValueText.setText(String.valueOf(currentValue));
				if (isMouthChosen) {
					cascadeMouthDetector.setSizeManuallyChanged(true);
					cascadeMouthDetector.setmRelativeMinObjectSize(currentValue);
				} else if (isNoseChosen) {
					cascadeNoseDetector.setSizeManuallyChanged(true);
					cascadeNoseDetector.setmRelativeMinObjectSize(currentValue);
				}
			}

			@Override
			public void onStartTrackingTouch(SeekBar arg0) {
			}

			@Override
			public void onStopTrackingTouch(SeekBar arg0) {
			}
		});

		maxObjectSeekBar = (VerticalSeekBar) findViewById(R.id.maxObjectSeekBar);
		verticalSeekBars.put("maxObject", maxObjectSeekBar);
		maxObjectValueText = (TextView) findViewById(R.id.maxObjectValueText);
		verticalSeekBarsTextValues.put("maxObject", maxObjectValueText);
		verticalSeekBarsTextNames.put("maxObject", (TextView) findViewById(R.id.maxObjectNameText));
		if (isMouthChosen) {
			maxObjectValueText.setText(String.valueOf(cascadeMouthDetector.getmRelativeMaxObjectSize()));
			maxObjectSeekBar.setProgress((int) (cascadeMouthDetector.getmRelativeMaxObjectSize() * 100));
		} else if (isNoseChosen) {
			maxObjectValueText.setText(String.valueOf(cascadeNoseDetector.getmRelativeMaxObjectSize()));
			maxObjectSeekBar.setProgress((int) (cascadeNoseDetector.getmRelativeMaxObjectSize() * 100));
		}
		maxObjectSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float currentValue = (float) ((float) progress / 100.0);
				float minObject = Float.parseFloat(minObjectValueText.getText().toString());
				if (currentValue < minObject) {
					currentValue = minObject;
				}
				maxObjectValueText.setText(String.valueOf(currentValue));
				if (isMouthChosen) {
					cascadeMouthDetector.setSizeManuallyChanged(true);
					cascadeMouthDetector.setmRelativeMaxObjectSize(currentValue);
				} else if (isNoseChosen) {
					cascadeNoseDetector.setSizeManuallyChanged(true);
					cascadeNoseDetector.setmRelativeMaxObjectSize(currentValue);
				}
			}
		});

		clipLimitSeekBar = (VerticalSeekBar) findViewById(R.id.clipLimitSeekBar);
		verticalSeekBars.put("clipLimit", clipLimitSeekBar);
		clipLimitValueText = (TextView) findViewById(R.id.clipLimitValueText);
		verticalSeekBarsTextValues.put("clipLimit", clipLimitValueText);
		verticalSeekBarsTextNames.put("clipLimit", (TextView) findViewById(R.id.clipLimitNameText));
		clipLimitValueText.setText(String.valueOf(claheAlgorithm.getCurrentClipLimit()));
		clipLimitSeekBar.setProgress((int) (claheAlgorithm.getCurrentClipLimit() * 100 / DetectorConstants.MAX_CLIP_LIMIT));
		clipLimitSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_CLIP_LIMIT - DetectorConstants.MIN_CLIP_LIMIT)) / 100)
						+ DetectorConstants.MIN_CLIP_LIMIT;
				clipLimitValueText.setText(String.valueOf(currentValue));
				claheAlgorithm.setCurrentClipLimit(currentValue);
			}
		});

		tileSizeSeekBar = (VerticalSeekBar) findViewById(R.id.tilesSizeSeekBar);
		verticalSeekBars.put("tileSize", tileSizeSeekBar);
		tileSizeValueText = (TextView) findViewById(R.id.tilesSizeValueText);
		verticalSeekBarsTextValues.put("tileSize", tileSizeValueText);
		verticalSeekBarsTextNames.put("tileSize", (TextView) findViewById(R.id.tilesSizeNameText));
		tileSizeValueText.setText(String.valueOf(claheAlgorithm.getCurrentTileSize()));
		tileSizeSeekBar.setProgress((int) (claheAlgorithm.getCurrentTileSize() * 100 / DetectorConstants.MAX_TILE_SIZE));
		tileSizeSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_TILE_SIZE - DetectorConstants.MIN_TILE_SIZE)) / 100)
						+ DetectorConstants.MIN_TILE_SIZE;
				tileSizeValueText.setText(String.valueOf(currentValue));
				claheAlgorithm.setCurrentTileSize(currentValue);
			}
		});

		gaussSeekBar = (VerticalSeekBar) findViewById(R.id.gaussSeekBar);
		verticalSeekBars.put("gaussSize", gaussSeekBar);
		gaussValueText = (TextView) findViewById(R.id.gaussValueText);
		verticalSeekBarsTextValues.put("gaussSize", gaussValueText);
		verticalSeekBarsTextNames.put("gaussSize", (TextView) findViewById(R.id.gaussNameText));
		gaussValueText.setText(String.valueOf(gaussSize));
		gaussSeekBar.setProgress((int) (gaussSize * 100 / DetectorConstants.MAX_GAUSS_SIZE));
		gaussSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_GAUSS_SIZE - DetectorConstants.MIN_GAUSS_SIZE)) / 100)
						+ DetectorConstants.MIN_GAUSS_SIZE;
				if (currentValue % 2 == 0) {
					currentValue += 1;
				}
				gaussValueText.setText(String.valueOf(currentValue));
				gaussSize = currentValue;
				drowsinessDetector.setGaussianBlur(gaussSize);
			}
		});

		blockSizeSeekBar = (VerticalSeekBar) findViewById(R.id.blockSizeSeekBar);
		verticalSeekBars.put("blockSize", blockSizeSeekBar);
		blockSizeText = (TextView) findViewById(R.id.blockSizeValueText);
		verticalSeekBarsTextValues.put("blockSize", blockSizeText);
		verticalSeekBarsTextNames.put("blockSize", (TextView) findViewById(R.id.blockSizeNameText));
		blockSizeText.setText(String.valueOf(binarizationAlgorithm.getBlockSize()));
		blockSizeSeekBar.setProgress((int) (binarizationAlgorithm.getBlockSize() * 100 / DetectorConstants.MAX_BLOCK_SIZE));
		blockSizeSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_BLOCK_SIZE - DetectorConstants.MIN_BLOCK_SIZE)) / 100)
						+ DetectorConstants.MIN_BLOCK_SIZE;
				if (currentValue % 2 == 0) {
					currentValue += 1;
				}
				blockSizeText.setText(String.valueOf(currentValue));
				binarizationAlgorithm.setBlockSize(currentValue);
			}
		});

		cSeekBar = (VerticalSeekBar) findViewById(R.id.cSeekBar);
		verticalSeekBars.put("C", cSeekBar);
		cText = (TextView) findViewById(R.id.cValueText);
		verticalSeekBarsTextValues.put("C", cText);
		verticalSeekBarsTextNames.put("C", (TextView) findViewById(R.id.cNameText));
		cText.setText(String.valueOf(binarizationAlgorithm.getC()));
		cSeekBar.setProgress((int) (Math.abs(binarizationAlgorithm.getC()) / (DetectorConstants.MAX_C_SIZE - DetectorConstants.MIN_C_SIZE)) * 100);
		cSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				float currentValue = (float) (((progress * (DetectorConstants.MAX_C_SIZE - DetectorConstants.MIN_C_SIZE)) / 100.0) + DetectorConstants.MIN_C_SIZE);
				cText.setText(String.valueOf(currentValue));
				binarizationAlgorithm.setC(currentValue);
			}
		});

		thresholdSeekBar = (VerticalSeekBar) findViewById(R.id.thresholdSeekBar);
		verticalSeekBars.put("threshold", thresholdSeekBar);
		thresholdText = (TextView) findViewById(R.id.thresholdValueText);
		verticalSeekBarsTextValues.put("threshold", thresholdText);
		verticalSeekBarsTextNames.put("threshold", (TextView) findViewById(R.id.thresholdNameText));
		thresholdText.setText(String.valueOf(binarizationAlgorithm.getThreshold()));
		thresholdSeekBar.setProgress((int) (binarizationAlgorithm.getThreshold() * 100 / DetectorConstants.MAX_THRESHOLD));
		thresholdSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_THRESHOLD - DetectorConstants.MIN_THRESHOLD)) / 100)
						+ DetectorConstants.MIN_THRESHOLD;
				thresholdText.setText(String.valueOf(currentValue));
				binarizationAlgorithm.setThreshold(currentValue);
			}
		});

		firstThresholdSeekBar = (VerticalSeekBar) findViewById(R.id.firstThresholdSeekBar);
		verticalSeekBars.put("firstThreshold", firstThresholdSeekBar);
		firstThresholdText = (TextView) findViewById(R.id.firstThresholdValueText);
		verticalSeekBarsTextValues.put("firstThreshold", firstThresholdText);
		verticalSeekBarsTextNames.put("firstThreshold", (TextView) findViewById(R.id.firstThresholdNameText));
		firstThresholdText.setText(String.valueOf(edgeDetectionAlgorithm.getFirstThreshold()));
		firstThresholdSeekBar.setProgress((int) (edgeDetectionAlgorithm.getFirstThreshold() * 100 / DetectorConstants.MAX_THRESHOLD));
		firstThresholdSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_THRESHOLD - DetectorConstants.MIN_THRESHOLD)) / 100)
						+ DetectorConstants.MIN_THRESHOLD;
				firstThresholdText.setText(String.valueOf(currentValue));
				edgeDetectionAlgorithm.setFirstThreshold(currentValue);
			}
		});

		secondThresholdSeekBar = (VerticalSeekBar) findViewById(R.id.secondThresholdSeekBar);
		verticalSeekBars.put("secondThreshold", secondThresholdSeekBar);
		secondThresholdText = (TextView) findViewById(R.id.secondThresholdValueText);
		verticalSeekBarsTextValues.put("secondThreshold", secondThresholdText);
		verticalSeekBarsTextNames.put("secondThreshold", (TextView) findViewById(R.id.secondThresholdNameText));
		secondThresholdText.setText(String.valueOf(edgeDetectionAlgorithm.getSecondThreshold()));
		secondThresholdSeekBar.setProgress((int) (edgeDetectionAlgorithm.getSecondThreshold() * 100 / DetectorConstants.MAX_THRESHOLD));
		secondThresholdSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_THRESHOLD - DetectorConstants.MIN_THRESHOLD)) / 100)
						+ DetectorConstants.MIN_THRESHOLD;
				secondThresholdText.setText(String.valueOf(currentValue));
				edgeDetectionAlgorithm.setFirstThreshold(currentValue);
			}
		});

		apertureSizeSeekBar = (VerticalSeekBar) findViewById(R.id.apertureSizeSeekBar);
		verticalSeekBars.put("apertureSize", apertureSizeSeekBar);
		apertureSizeText = (TextView) findViewById(R.id.apertureSizeValueText);
		verticalSeekBarsTextValues.put("apertureSize", apertureSizeText);
		verticalSeekBarsTextNames.put("apertureSize", (TextView) findViewById(R.id.apertureSizeNameText));
		apertureSizeText.setText(String.valueOf(edgeDetectionAlgorithm.getApertureSize()));
		apertureSizeSeekBar.setProgress((int) (edgeDetectionAlgorithm.getApertureSize() * 100 / DetectorConstants.MAX_APERTURE_SIZE));
		apertureSizeSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_APERTURE_SIZE - DetectorConstants.MIN_APERTURE_SIZE)) / 100)
						+ DetectorConstants.MIN_APERTURE_SIZE;
				if (currentValue % 2 == 0) {
					currentValue += 1;
				}
				apertureSizeText.setText(String.valueOf(currentValue));
				edgeDetectionAlgorithm.setApertureSize(currentValue);
			}
		});

		erosionSizeSeekBar = (VerticalSeekBar) findViewById(R.id.erosionSeekBar);
		verticalSeekBars.put("erosionSize", erosionSizeSeekBar);
		erosionSizeText = (TextView) findViewById(R.id.erosionValueText);
		verticalSeekBarsTextValues.put("erosionSize", erosionSizeText);
		verticalSeekBarsTextNames.put("erosionSize", (TextView) findViewById(R.id.erosionNameText));
		erosionSizeText.setText(String.valueOf(darkBrightRatioAlgorithm.getErosionSize()));
		erosionSizeSeekBar.setProgress((int) (darkBrightRatioAlgorithm.getErosionSize() * 100 / DetectorConstants.MAX_EROSION_SIZE));
		erosionSizeSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
				int currentValue = ((progress * (DetectorConstants.MAX_EROSION_SIZE - DetectorConstants.MIN_EROSION_SIZE)) / 100)
						+ DetectorConstants.MIN_EROSION_SIZE;
				erosionSizeText.setText(String.valueOf(currentValue));
				darkBrightRatioAlgorithm.setErosionSize(currentValue);
			}
		});
	}

	private void isRawDataExists(Context context) {
		try {
			File internalDir = context.getDir("stasm", Context.MODE_PRIVATE);
			Log.i(TAG, "files for stasm: " + internalDir.getAbsolutePath());
			File frontalface_xml = new File(internalDir, "haarcascade_frontalface_alt2.xml");
			File lefteye_xml = new File(internalDir, "haarcascade_mcs_lefteye.xml");
			File righteye_xml = new File(internalDir, "haarcascade_mcs_righteye.xml");
			File mounth_xml = new File(internalDir, "haarcascade_mcs_mouth.xml");

			if (frontalface_xml.exists() && lefteye_xml.exists() && righteye_xml.exists() && mounth_xml.exists()) {
				Log.i(TAG, "RawDataExists");
			} else {
				copyRawDataToInternal(context, R.raw.haarcascade_frontalface_alt2, frontalface_xml);
				copyRawDataToInternal(context, R.raw.haarcascade_mcs_lefteye, lefteye_xml);
				copyRawDataToInternal(context, R.raw.haarcascade_mcs_righteye, righteye_xml);
				copyRawDataToInternal(context, R.raw.haarcascade_mcs_mouth, mounth_xml);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void copyRawDataToInternal(Context context, int id, File file) {
		Log.i(TAG, "copyRawDataToInternal: " + file.toString());
		try {
			InputStream is = context.getResources().openRawResource(id);
			FileOutputStream fos = new FileOutputStream(file);

			int data;
			byte[] buffer = new byte[4096];
			while ((data = is.read(buffer)) != -1) {
				fos.write(buffer, 0, data);
			}
			is.close();
			fos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		Log.i(TAG, "copyRawDataToInternal done");
	}

	public native void FindFace(long matAddrGr, long matAddrRgba);

	public native void FindEyes(long matAddrGr, long matAddrRgba);

	public native void PrepareFindFace(String fileName);

	public native void PrepareFindEyes(String fileName);

	public native void ApplyCLAHE(long matAddrSrc, double clipLimit);

	public native void ApplyCLAHEExt(long matAddrSrc, double clipLimit, double tileWidth, double tileHeight);

	public native void SegmentSkin(long matAddrSrc, long matAddrDst);

	public native void FindFeatures(long matAddrGray, long matAddrRgb);

	public native void FindCornerHarris(long matAddrGray, long matAddrDst);

	public native void CalculateOpticalFlow(long matAddrSrc, long matAddrDst);

	public native int[] FindFaceLandmarks(long matAddrGr);

	public native int[] AddFaceLandmarks(long matAddrGr, float[] points);

}
