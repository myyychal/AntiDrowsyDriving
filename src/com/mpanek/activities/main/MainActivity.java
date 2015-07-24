package com.mpanek.activities.main;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;

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
import android.content.Context;
import android.hardware.Camera.Face;
import android.hardware.Camera.FaceDetectionListener;
import android.hardware.Camera;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.text.Layout;
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
import android.widget.PopupMenu;
import android.widget.PopupMenu.OnMenuItemClickListener;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VerticalSeekBar;

import com.mpanek.constants.DetectorConstants;
import com.mpanek.constants.DrawingConstants;
import com.mpanek.constants.ViewModesConstants;
import com.mpanek.detection.SnapdragonFacialFeaturesDetector;
import com.mpanek.detection.eyes.CascadeEyesDetector;
import com.mpanek.detection.face.CascadeFaceDetector;
import com.mpanek.detection.face.ColorSegmentationFaceDetector;
import com.mpanek.detection.mouth.CascadeMouthDetector;
import com.mpanek.detection.nose.CascadeNoseDetector;
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

	private int mCameraId = CameraBridgeViewBase.CAMERA_ID_FRONT;

	private int mViewMode;
	private int mEqHistMode;

	private Mat mRgba;
	private Mat mGray;
	private Mat currentlyUsedFrame;

	private Mat matOpFlowPrev, matOpFlowThis;
	private MatOfPoint MOPcorners;
	private MatOfPoint2f mMOP2f1, mMOP2f2, mMOP2fptsPrev, mMOP2fptsThis,
			mMOP2fptsSafe;
	private MatOfByte mMOBStatus;
	private MatOfFloat mMOFerr;
	private List<Point> cornersThis, cornersPrev;
	private List<Byte> byteStatus;
	private int iGFFTMax = 40;

	private SubMenu mItemPreviewEqHistMenu;
	private SubMenu mItemPreviewFindFaceMenu;
	private SubMenu mItemPreviewFindEyesMenu;
	private SubMenu mItemPreviewFindMouthMenu;
	private SubMenu mItemPreviewOtherMenu;

	private List<android.hardware.Camera.Size> mResolutionList;
	private MenuItem[] mResolutionMenuItems;

	private CustomCameraView mCustomCameraView;

	SnapdragonFacialFeaturesDetector snapdragonFacialFeaturesDetector;
	CascadeFaceDetector cascadeFaceDetector;
	CascadeEyesDetector cascadeEyesDetector;
	CascadeMouthDetector cascadeMouthDetector;
	CascadeNoseDetector cascadeNoseDetector;
	ColorSegmentationFaceDetector colorSegmentationFaceDetector;

	OrientationEventListener orientationEventListener;
	int deviceOrientation;

	private static boolean isMouthWithEyes = true;
	private static boolean isFaceTracking = true;
	private static boolean isCurrentFrameRgb = true;

	private static boolean isGuiHidden = false;

	private static boolean isNoseChosen = false;
	private static boolean isMouthChosen = false;

	private int currentClipLimit = 4;
	private int currentTileSize = 4;

	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");

				System.loadLibrary("anti_drowsy_driving");

				int faceResourceId = getResources().getIdentifier(
						"lbpcascade_frontalface", "raw", getPackageName());
				InputStream is = getResources().openRawResource(faceResourceId);
				File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
				File faceFile = cascadeFaceDetector.prepare(is, cascadeDir);
				PrepareFindFace(faceFile.getAbsolutePath());

				int eyesResourceId = getResources().getIdentifier(
						"haarcascade_eye_tree_eyeglasses", "raw",
						getPackageName());
				is = getResources().openRawResource(eyesResourceId);
				File eyesFile = cascadeEyesDetector.prepare(is, cascadeDir);
				PrepareFindEyes(eyesFile.getAbsolutePath());

				int mouthResourceId = getResources().getIdentifier(
						"haarcascade_mcs_mouth", "raw", getPackageName());
				is = getResources().openRawResource(mouthResourceId);
				cascadeMouthDetector.prepare(is, cascadeDir);

				int noseResourceId = getResources().getIdentifier(
						"haarcascade_mcs_nose", "raw", getPackageName());
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

		snapdragonFacialFeaturesDetector = new SnapdragonFacialFeaturesDetector(
				((WindowManager) getSystemService(Context.WINDOW_SERVICE))
						.getDefaultDisplay(),
				TAG, mCameraId);

		cascadeFaceDetector = new CascadeFaceDetector();
		cascadeEyesDetector = new CascadeEyesDetector();
		cascadeMouthDetector = new CascadeMouthDetector();
		cascadeNoseDetector = new CascadeNoseDetector();

		colorSegmentationFaceDetector = new ColorSegmentationFaceDetector();

		initVerticalSeekBars();
	}

	private void startOrientationListener() {
		orientationEventListener = new OrientationEventListener(this,
				SensorManager.SENSOR_DELAY_NORMAL) {
			@Override
			public void onOrientationChanged(int orientation) {
				deviceOrientation = orientation;
				Log.i(TAG, "Device orientation: " + deviceOrientation);
				Log.i(TAG,
						"Present orientation: "
								+ (90 * Math.round(deviceOrientation / 90))
								% 360);
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
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this,
				mLoaderCallback);
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
//			Core.flip(mRgba, mRgba, 1);
//			Core.flip(mGray, mGray, 1);
		}

		if (isCurrentFrameRgb) {
			currentlyUsedFrame = mRgba;
		} else {
			currentlyUsedFrame = mGray;
		}

		final int viewMode = mViewMode;
		final int eqHistMode = mEqHistMode;

		switch (eqHistMode) {
		case ViewModesConstants.VIEW_MODE_EQ_NONE:
			break;

		case ViewModesConstants.VIEW_MODE_EQ_HIST_CPP:
			EqualizeHistogram(currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_EQ_HIST_CLAHE_CPP:
			ApplyCLAHEExt(currentlyUsedFrame.getNativeObjAddr(),
					(double) currentClipLimit, (double) currentTileSize,
					(double) currentTileSize);
			break;
		}

		switch (viewMode) {

		case ViewModesConstants.VIEW_MODE_NONE:
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_SNAPDRAGON:
			snapdragonFacialFeaturesDetector.findFace(mRgba, this
					.getResources().getConfiguration().orientation);
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_JAVA:
			Rect face = cascadeFaceDetector.findFace(currentlyUsedFrame);
			if (face != null) {
				DrawingUtils.drawRect(face, currentlyUsedFrame,
						DrawingConstants.FACE_RECT_COLOR);
			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_CPP:
			FindFace(currentlyUsedFrame.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_JAVA:
			currentlyUsedFrame = colorSegmentationFaceDetector
					.detectFaceYCrCb(currentlyUsedFrame);
			break;

		case ViewModesConstants.VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_CPP:
			SegmentSkin(mRgba.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_JAVA:
			Rect foundFace;
			if (isFaceTracking) {
				foundFace = cascadeFaceDetector.findFace(mGray);
			} else {
				foundFace = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFace != null) {
				DrawingUtils.drawRect(foundFace, currentlyUsedFrame,
						DrawingConstants.FACE_RECT_COLOR);
				Rect[] eyes = cascadeEyesDetector.findEyes(mGray, foundFace);
				DrawingUtils.drawRects(eyes, currentlyUsedFrame,
						DrawingConstants.EYES_RECT_COLOR);
			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_CPP:
			FindEyes(currentlyUsedFrame.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_FIND_MOUTH_CASCADE_JAVA:
			Rect foundFace2;
			if (isFaceTracking) {
				foundFace2 = cascadeFaceDetector.findFace(currentlyUsedFrame);
			} else {
				foundFace2 = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFace2 != null) {
				DrawingUtils.drawRect(foundFace2, currentlyUsedFrame,
						DrawingConstants.FACE_RECT_COLOR);
				if (isMouthWithEyes) {
					Rect[] eyes = cascadeEyesDetector.findEyes(
							currentlyUsedFrame, foundFace2);
					DrawingUtils.drawRects(eyes, currentlyUsedFrame,
							DrawingConstants.EYES_RECT_COLOR);
				}
				Rect[] mouths = cascadeMouthDetector.findMouth(
						currentlyUsedFrame, foundFace2);
				DrawingUtils.drawRects(mouths, currentlyUsedFrame,
						DrawingConstants.MOUTH_RECT_COLOR);
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
				DrawingUtils.drawRect(foundFace3, currentlyUsedFrame,
						DrawingConstants.FACE_RECT_COLOR);
				Rect[] noses = cascadeNoseDetector.findNose(currentlyUsedFrame,
						foundFace3);
				DrawingUtils.drawRects(noses, currentlyUsedFrame,
						DrawingConstants.NOSE_RECT_COLOR);
			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_JAVA:
			MatOfPoint cornersPoints = new MatOfPoint();
			Imgproc.goodFeaturesToTrack(mGray, cornersPoints, 100, 0.01, 10);
			int rows = cornersPoints.rows();
			List<Point> listOfPoints = cornersPoints.toList();
			for (int x = 0; x < rows; x++) {
				Core.circle(currentlyUsedFrame, listOfPoints.get(x), 5,
						DrawingConstants.RED);
			}
			break;

		case ViewModesConstants.VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_CPP:
			FindFeatures(mGray.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_FIND_CORNER_HARRIS_JAVA:
			Mat dst = Mat.zeros(mGray.size(), CvType.CV_32FC1);
			Mat dst_norm = new Mat();
			Mat dst_norm_scaled = new Mat();

			int blockSize = 2;
			int apertureSize = 3;
			double k = 0.04;

			Imgproc.cornerHarris(mGray, dst, blockSize, apertureSize, k,
					Imgproc.BORDER_DEFAULT);
			Core.normalize(dst, dst_norm, 0, 255, Core.NORM_MINMAX,
					CvType.CV_32FC1, new Mat());
			Core.convertScaleAbs(dst_norm, dst_norm_scaled);

			currentlyUsedFrame = dst_norm_scaled;
			break;

		case ViewModesConstants.VIEW_MODE_FIND_CORNER_HARRIS_CPP:
			FindCornerHarris(mGray.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_OPTICAL_FLOW_JAVA:
			if (mMOP2fptsPrev.rows() == 0) {

				Imgproc.cvtColor(mRgba, matOpFlowThis, Imgproc.COLOR_RGBA2GRAY);

				matOpFlowThis.copyTo(matOpFlowPrev);

				Imgproc.goodFeaturesToTrack(matOpFlowPrev, MOPcorners,
						iGFFTMax, 0.05, 20);
				mMOP2fptsPrev.fromArray(MOPcorners.toArray());

				mMOP2fptsPrev.copyTo(mMOP2fptsSafe);
			} else {

				matOpFlowThis.copyTo(matOpFlowPrev);

				Imgproc.cvtColor(mRgba, matOpFlowThis, Imgproc.COLOR_RGBA2GRAY);

				Imgproc.goodFeaturesToTrack(matOpFlowThis, MOPcorners,
						iGFFTMax, 0.05, 20);
				mMOP2fptsThis.fromArray(MOPcorners.toArray());

				mMOP2fptsSafe.copyTo(mMOP2fptsPrev);

				mMOP2fptsThis.copyTo(mMOP2fptsSafe);
			}

			Video.calcOpticalFlowPyrLK(matOpFlowPrev, matOpFlowThis,
					mMOP2fptsPrev, mMOP2fptsThis, mMOBStatus, mMOFerr);

			cornersPrev = mMOP2fptsPrev.toList();
			cornersThis = mMOP2fptsThis.toList();
			byteStatus = mMOBStatus.toList();

			int y = byteStatus.size() - 1;

			for (int x = 0; x < y; x++) {
				if (byteStatus.get(x) == 1) {
					Point pt = cornersThis.get(x);
					Point pt2 = cornersPrev.get(x);
					Core.circle(currentlyUsedFrame, pt, 5,
							DrawingConstants.RED, 5);
					Core.line(currentlyUsedFrame, pt, pt2,
							DrawingConstants.RED, 3);
				}
			}

			break;

		case ViewModesConstants.VIEW_MODE_OPTICAL_FLOW_CPP:
			CalculateOpticalFlow(mRgba.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case ViewModesConstants.VIEW_MODE_START_DROWSINESS_DETECTION:
			currentlyUsedFrame = mGray;
			ApplyCLAHEExt(mGray.getNativeObjAddr(), (double) currentClipLimit,
					(double) currentTileSize, (double) currentTileSize);
			Rect foundFaceInDetection;
			if (isFaceTracking) {
				Rect boundingBox = new Rect(0, 0, mGray.width(), mGray.height());
				double boundingMultiplier = 0.1;
				boundingBox.x += boundingMultiplier * mGray.width();
				boundingBox.width -= 2 * boundingMultiplier * mGray.width();
				foundFaceInDetection = cascadeFaceDetector.findFace(mGray,
						boundingBox);
			} else {
				foundFaceInDetection = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFaceInDetection != null) {
				Rect foundFaceForEyes = foundFaceInDetection.clone();
				foundFaceForEyes.height /= 2;
				foundFaceForEyes.y = (int) (foundFaceForEyes.y + 0.1 * mGray
						.height());
				Rect[] eyes = cascadeEyesDetector.findEyes(mGray,
						foundFaceForEyes);

				Rect foundFaceForMouth = foundFaceInDetection.clone();
				foundFaceForMouth.height /= 2;
				foundFaceForMouth.y = foundFaceForMouth.y
						+ foundFaceForMouth.height;
				Rect mouths[] = cascadeMouthDetector.findMouth(mGray,
						foundFaceForMouth);

				DrawingUtils.drawRect(foundFaceInDetection, currentlyUsedFrame,
						DrawingConstants.FACE_RECT_COLOR);
				DrawingUtils.drawRects(eyes, currentlyUsedFrame,
						DrawingConstants.EYES_RECT_COLOR);
				DrawingUtils.drawRects(mouths, currentlyUsedFrame,
						DrawingConstants.MOUTH_RECT_COLOR);
			}
			break;

		}

		return currentlyUsedFrame;
	}

	public boolean onOptionsItemSelected(MenuItem item) {
		if (item.getGroupId() == 1) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_SNAPDRAGON;
				break;
			case 1:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_JAVA;
				break;
			case 2:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_CPP;
				break;
			case 3:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_JAVA;
				break;
			case 4:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_CPP;
				break;
			}
		} else if (item.getGroupId() == 2) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_JAVA;
				break;
			case 1:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_CPP;
				break;
			}
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
		} else if (item.getGroupId() == 4) {
			switch (item.getItemId()) {
			case 0:
				mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_NONE;
				break;
			case 1:
				mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_HIST_CPP;
				break;
			case 2:
				mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_HIST_CLAHE_CPP;
				break;
			}
		} else if (item.getGroupId() == 5) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_JAVA;
				break;
			case 1:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_CPP;
				break;
			case 2:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_CORNER_HARRIS_JAVA;
				break;
			case 3:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_CORNER_HARRIS_CPP;
				break;
			case 4:
				mViewMode = ViewModesConstants.VIEW_MODE_OPTICAL_FLOW_JAVA;
				break;
			case 5:
				mViewMode = ViewModesConstants.VIEW_MODE_OPTICAL_FLOW_CPP;
				break;
			}
		} else if (item.getGroupId() == 6) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_NOSE_CASCADE_JAVA;
				break;
			}
		}
		return true;
	}
	
	public void showOptionsMenu(View v) {
		openOptionsMenu();
	}

	public void showCascadeFaceFilesMenu(final View v) {
		PopupMenu popup = new PopupMenu(this, v);
		MenuInflater inflater = popup.getMenuInflater();
		inflater.inflate(R.menu.cascade_face_files_menu, popup.getMenu());
		popup.setOnMenuItemClickListener(new OnMenuItemClickListener() {
			@Override
			public boolean onMenuItemClick(MenuItem menuitem) {
				int faceResourceId = getResources()
						.getIdentifier(menuitem.getTitle().toString(), "raw",
								getPackageName());
				InputStream is = getResources().openRawResource(faceResourceId);
				File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
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
		Toast.makeText(this,
				"Face tracking: " + String.valueOf(isFaceTracking),
				Toast.LENGTH_SHORT).show();
	}

	public void showCascadeEyesFilesMenu(final View v) {
		PopupMenu popup = new PopupMenu(this, v);
		MenuInflater inflater = popup.getMenuInflater();
		inflater.inflate(R.menu.cascade_eyes_files_menu, popup.getMenu());
		popup.setOnMenuItemClickListener(new OnMenuItemClickListener() {
			@Override
			public boolean onMenuItemClick(MenuItem menuitem) {
				int eyesResourceId = getResources()
						.getIdentifier(menuitem.getTitle().toString(), "raw",
								getPackageName());
				InputStream is = getResources().openRawResource(eyesResourceId);
				File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
				File faceFile = cascadeEyesDetector.prepare(is, cascadeDir);
				PrepareFindFace(faceFile.getAbsolutePath());
				return false;
			}
		});
		popup.show();
	}

	public void changeResolution(final View v) {
		PopupMenu popup = new PopupMenu(this, v);
		mResolutionList = mCustomCameraView.getResolutionList();
		mResolutionMenuItems = new MenuItem[mResolutionList.size()];

		ListIterator<android.hardware.Camera.Size> resolutionItr = mResolutionList
				.listIterator();
		int idx = 0;
		while (resolutionItr.hasNext()) {
			android.hardware.Camera.Size element = resolutionItr.next();
			mResolutionMenuItems[idx] = popup.getMenu().add(
					2,
					idx,
					Menu.NONE,
					Integer.valueOf(element.width).toString() + "x"
							+ Integer.valueOf(element.height).toString());
			idx++;
		}

		popup.setOnMenuItemClickListener(new OnMenuItemClickListener() {

			@Override
			public boolean onMenuItemClick(MenuItem item) {
				int id = item.getItemId();
				android.hardware.Camera.Size resolution = mResolutionList
						.get(id);
				mCustomCameraView.setResolution(resolution);
				resolution = mCustomCameraView.getResolution();
				String caption = Integer.valueOf(resolution.width).toString()
						+ "x" + Integer.valueOf(resolution.height).toString();
				Toast.makeText(getBaseContext(), caption, Toast.LENGTH_SHORT)
						.show();
				return false;
			}
		});
		popup.show();
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
		mItemPreviewEqHistMenu = menu.addSubMenu("Equalize histogram");
		mItemPreviewFindFaceMenu = menu.addSubMenu("Find face");
		mItemPreviewFindEyesMenu = menu.addSubMenu("Find eyes");
		mItemPreviewFindMouthMenu = menu.addSubMenu("Find mouth");
		menu.add(6, 0, Menu.NONE, "Find nose");
		mItemPreviewOtherMenu = menu.addSubMenu("Other");

		mItemPreviewFindFaceMenu.add(1, 0, Menu.NONE, "Snapdragon");
		mItemPreviewFindFaceMenu.add(1, 1, Menu.NONE, "Haar (Java)");
		mItemPreviewFindFaceMenu.add(1, 2, Menu.NONE, "Haar (Cpp)");
		mItemPreviewFindFaceMenu.add(1, 3, Menu.NONE,
				"Color segmentation (Java)");
		mItemPreviewFindFaceMenu.add(1, 4, Menu.NONE,
				"Color segmentation (Cpp)");

		mItemPreviewFindEyesMenu.add(2, 0, Menu.NONE, "Java");
		mItemPreviewFindEyesMenu.add(2, 1, Menu.NONE, "Cpp");

		mItemPreviewFindMouthMenu.add(3, 0, Menu.NONE, "Without eyes");
		mItemPreviewFindMouthMenu.add(3, 1, Menu.NONE, "With eyes");

		mItemPreviewEqHistMenu.add(4, 0, Menu.NONE, "None");
		mItemPreviewEqHistMenu.add(4, 1, Menu.NONE, "Standard");
		mItemPreviewEqHistMenu.add(4, 2, Menu.NONE, "CLAHE");

		mItemPreviewOtherMenu.add(5, 0, Menu.NONE, "Find features (Java)");
		mItemPreviewOtherMenu.add(5, 1, Menu.NONE, "Find features (Cpp)");
		mItemPreviewOtherMenu.add(5, 2, Menu.NONE, "Find corner Harris (Java)");
		mItemPreviewOtherMenu.add(5, 3, Menu.NONE, "Find corner Harris (Cpp)");
		mItemPreviewOtherMenu.add(5, 4, Menu.NONE, "Optical flow (Java)");
		mItemPreviewOtherMenu.add(5, 5, Menu.NONE, "Optical flow (Cpp)");

		return true;
	}

	public void resetView(View v) {
		mViewMode = ViewModesConstants.VIEW_MODE_NONE;
		mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_NONE;
	}

	public void startDrowsinessDetection(View v) {
		switch (mViewMode) {
		case ViewModesConstants.VIEW_MODE_START_DROWSINESS_DETECTION:
			mViewMode = ViewModesConstants.VIEW_MODE_NONE;
			break;
		case ViewModesConstants.VIEW_MODE_NONE:
			mViewMode = ViewModesConstants.VIEW_MODE_START_DROWSINESS_DETECTION;
			break;
		}
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
	
	private void changeVisibiltyOfChildViews(View parentView, int visibiltyFlag){
		if (parentView instanceof ViewGroup){
			for (int i=0; i<((ViewGroup)parentView).getChildCount(); i++){
				changeVisibiltyOfChildViews(((ViewGroup)parentView).getChildAt(i), visibiltyFlag);
			}
		} else {
			if (!(parentView instanceof CustomCameraView
					|| parentView.getId() == R.id.showHideGui)) {
				parentView.setVisibility(visibiltyFlag);
			}
		}
	}

	public void chooseObjectToChangeItsSize(final View v) {
		PopupMenu popup = new PopupMenu(this, v);
		MenuInflater inflater = popup.getMenuInflater();
		inflater.inflate(R.menu.detection_things, popup.getMenu());
		popup.setOnMenuItemClickListener(new OnMenuItemClickListener() {
			@Override
			public boolean onMenuItemClick(MenuItem menuitem) {
				String menuItemTitle = menuitem.getTitle().toString();
				if (menuItemTitle.equals("Mouth")) {
					isMouthChosen = true;
					isNoseChosen = false;
					minObjectValueText.setText(String.valueOf(cascadeMouthDetector
							.getmRelativeMinObjectSize()));
					minObjectSeekBar.setProgress((int) (cascadeMouthDetector
							.getmRelativeMinObjectSize() * 100));
					maxObjectValueText.setText(String.valueOf(cascadeMouthDetector
							.getmRelativeMaxObjectSize()));
					maxObjectSeekBar.setProgress((int) (cascadeMouthDetector
							.getmRelativeMaxObjectSize() * 100));
					Button thisButton = (Button) v;
					thisButton.setText("Size: mouth");
				} else if (menuItemTitle.equals("Nose")) {
					isNoseChosen = true;
					isMouthChosen = false;
					minObjectValueText.setText(String.valueOf(cascadeNoseDetector
							.getmRelativeMinObjectSize()));
					minObjectSeekBar.setProgress((int) (cascadeNoseDetector
							.getmRelativeMinObjectSize() * 100));
					maxObjectValueText.setText(String.valueOf(cascadeNoseDetector
							.getmRelativeMaxObjectSize()));
					maxObjectSeekBar.setProgress((int) (cascadeNoseDetector
							.getmRelativeMaxObjectSize() * 100));
					Button thisButton = (Button) v;
					thisButton.setText("Size: nose");
				}
				return false;
			}
		});
		popup.show();
	}

	public void initVerticalSeekBars() {

		scaleFactorSeekBar = (VerticalSeekBar) findViewById(R.id.scaleFactorSeekBar);
		scaleFactorValueText = (TextView) findViewById(R.id.scaleFactorValueText);
		scaleFactorValueText.setText(String.valueOf(cascadeFaceDetector
				.getScaleFactor()));
		scaleFactorSeekBar.setProgress((int) ((cascadeFaceDetector
				.getScaleFactor() / DetectorConstants.MAX_SCALE_FACTOR) * 100));
		scaleFactorSeekBar
				.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
					@Override
					public void onStopTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onStartTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onProgressChanged(SeekBar seekBar,
							int progress, boolean fromUser) {
						float currentValue = ((progress * (DetectorConstants.MAX_SCALE_FACTOR - DetectorConstants.MIN_SCALE_FACTOR)) / 100)
								+ DetectorConstants.MIN_SCALE_FACTOR;
						scaleFactorValueText.setText(String.valueOf(MathUtils
								.round(currentValue, 2)));
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
		minNeighsValueText = (TextView) findViewById(R.id.minNeighboursValueText);
		minNeighsValueText.setText(String.valueOf(cascadeFaceDetector
				.getMinNeighbours()));
		minNeighsSeekBar.setProgress(cascadeFaceDetector.getMinNeighbours()
				* 100 / DetectorConstants.MAX_MIN_NEIGHBOURS);
		minNeighsSeekBar
				.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
					@Override
					public void onStopTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onStartTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onProgressChanged(SeekBar seekBar,
							int progress, boolean fromUser) {
						int currentValue = ((progress * (DetectorConstants.MAX_MIN_NEIGHBOURS - DetectorConstants.MIN_MIN_NEIGHBOURS)) / 100)
								+ DetectorConstants.MIN_MIN_NEIGHBOURS;
						minNeighsValueText.setText(String.valueOf(MathUtils
								.round(currentValue, 2)));
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
		minFaceValueText = (TextView) findViewById(R.id.minFaceValueText);
		minFaceValueText.setText(String.valueOf(cascadeFaceDetector
				.getmRelativeMinObjectSize()));
		minFaceSeekBar.setProgress((int) (cascadeFaceDetector
				.getmRelativeMinObjectSize() * 100));
		minFaceSeekBar
				.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
					@Override
					public void onStopTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onStartTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onProgressChanged(SeekBar seekBar,
							int progress, boolean fromUser) {
						float currentValue = (float) ((float) progress / 100.0);
						float maxFace = Float.parseFloat(maxFaceValueText
								.getText().toString());
						if (currentValue > maxFace) {
							currentValue = maxFace;
						}
						minFaceValueText.setText(String.valueOf(currentValue));
						cascadeFaceDetector.setSizeManuallyChanged(true);
						cascadeFaceDetector
								.setmRelativeMinObjectSize(currentValue);
					}
				});

		maxFaceSeekBar = (VerticalSeekBar) findViewById(R.id.maxFaceSeekBar);
		maxFaceValueText = (TextView) findViewById(R.id.maxFaceValueText);
		maxFaceValueText.setText(String.valueOf(cascadeFaceDetector
				.getmRelativeMaxObjectSize()));
		maxFaceSeekBar.setProgress((int) (cascadeFaceDetector
				.getmRelativeMaxObjectSize() * 100));
		maxFaceSeekBar
				.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
					@Override
					public void onStopTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onStartTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onProgressChanged(SeekBar seekBar,
							int progress, boolean fromUser) {
						float currentValue = (float) ((float) progress / 100.0);
						float minFace = Float.parseFloat(minFaceValueText
								.getText().toString());
						if (currentValue < minFace) {
							currentValue = minFace;
						}
						maxFaceValueText.setText(String.valueOf(currentValue));
						cascadeFaceDetector.setSizeManuallyChanged(true);
						cascadeFaceDetector
								.setmRelativeMaxObjectSize(currentValue);
					}
				});

		minEyeSeekBar = (VerticalSeekBar) findViewById(R.id.minEyeSeekBar);
		minEyeValueText = (TextView) findViewById(R.id.minEyeValueText);
		minEyeValueText.setText(String.valueOf(cascadeEyesDetector
				.getmRelativeMinObjectSize()));
		minEyeSeekBar.setProgress((int) (cascadeEyesDetector
				.getmRelativeMinObjectSize() * 100));
		minEyeSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress,
					boolean fromUser) {
				float currentValue = (float) ((float) progress / 100.0);
				float maxEye = Float.parseFloat(maxEyeValueText.getText()
						.toString());
				if (currentValue > maxEye) {
					currentValue = maxEye;
				}
				minEyeValueText.setText(String.valueOf(currentValue));
				cascadeEyesDetector.setSizeManuallyChanged(true);
				cascadeEyesDetector.setmRelativeMinObjectSize(currentValue);
			}
		});

		maxEyeSeekBar = (VerticalSeekBar) findViewById(R.id.maxEyeSeekBar);
		maxEyeValueText = (TextView) findViewById(R.id.maxEyeValueText);
		maxEyeValueText.setText(String.valueOf(cascadeEyesDetector
				.getmRelativeMaxObjectSize()));
		maxEyeSeekBar.setProgress((int) (cascadeEyesDetector
				.getmRelativeMaxObjectSize() * 100));
		maxEyeSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress,
					boolean fromUser) {
				float currentValue = (float) ((float) progress / 100.0);
				float minEye = Float.parseFloat(minEyeValueText.getText()
						.toString());
				if (currentValue < minEye) {
					currentValue = minEye;
				}
				maxEyeValueText.setText(String.valueOf(currentValue));
				cascadeEyesDetector.setSizeManuallyChanged(true);
				cascadeEyesDetector.setmRelativeMaxObjectSize(currentValue);
			}
		});

		minObjectSeekBar = (VerticalSeekBar) findViewById(R.id.minObjectSeekBar);
		minObjectValueText = (TextView) findViewById(R.id.minObjectValueText);
		if (isMouthChosen) {
			minObjectValueText.setText(String.valueOf(cascadeMouthDetector
					.getmRelativeMinObjectSize()));
			minObjectSeekBar.setProgress((int) (cascadeMouthDetector
					.getmRelativeMinObjectSize() * 100));
		} else if (isNoseChosen) {
			minObjectValueText.setText(String.valueOf(cascadeNoseDetector
					.getmRelativeMinObjectSize()));
			minObjectSeekBar.setProgress((int) (cascadeNoseDetector
					.getmRelativeMinObjectSize() * 100));
		}
		minObjectSeekBar
				.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
					@Override
					public void onProgressChanged(SeekBar seekBar,
							int progress, boolean fromUser) {
						float currentValue = (float) ((float) progress / 100.0);
						float maxObject = Float.parseFloat(maxObjectValueText
								.getText().toString());
						if (currentValue > maxObject) {
							currentValue = maxObject;
						}
						minObjectValueText.setText(String.valueOf(currentValue));
						if (isMouthChosen) {
							cascadeMouthDetector.setSizeManuallyChanged(true);
							cascadeMouthDetector
									.setmRelativeMinObjectSize(currentValue);
						} else if (isNoseChosen) {
							cascadeNoseDetector.setSizeManuallyChanged(true);
							cascadeNoseDetector
									.setmRelativeMinObjectSize(currentValue);
						}
					}

					@Override
					public void onStartTrackingTouch(SeekBar arg0) {
						// TODO Auto-generated method stub

					}

					@Override
					public void onStopTrackingTouch(SeekBar arg0) {
						// TODO Auto-generated method stub

					}
				});

		maxObjectSeekBar = (VerticalSeekBar) findViewById(R.id.maxObjectSeekBar);
		maxObjectValueText = (TextView) findViewById(R.id.maxObjectValueText);
		if (isMouthChosen) {
			maxObjectValueText.setText(String.valueOf(cascadeMouthDetector
					.getmRelativeMaxObjectSize()));
			maxObjectSeekBar.setProgress((int) (cascadeMouthDetector
					.getmRelativeMaxObjectSize() * 100));
		} else if (isNoseChosen) {
			maxObjectValueText.setText(String.valueOf(cascadeNoseDetector
					.getmRelativeMaxObjectSize()));
			maxObjectSeekBar.setProgress((int) (cascadeNoseDetector
					.getmRelativeMaxObjectSize() * 100));
		}
		maxObjectSeekBar.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
			@Override
			public void onStopTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onStartTrackingTouch(SeekBar seekBar) {
			}

			@Override
			public void onProgressChanged(SeekBar seekBar, int progress,
					boolean fromUser) {
				float currentValue = (float) ((float) progress / 100.0);
				float minObject = Float.parseFloat(minObjectValueText.getText()
						.toString());
				if (currentValue < minObject) {
					currentValue = minObject;
				}
				maxObjectValueText.setText(String.valueOf(currentValue));
				if (isMouthChosen) {
					cascadeMouthDetector.setSizeManuallyChanged(true);
					cascadeMouthDetector
							.setmRelativeMaxObjectSize(currentValue);
				} else if (isNoseChosen) {
					cascadeNoseDetector.setSizeManuallyChanged(true);
					cascadeNoseDetector.setmRelativeMaxObjectSize(currentValue);
				}
			}
		});

		clipLimitSeekBar = (VerticalSeekBar) findViewById(R.id.clipLimitSeekBar);
		clipLimitValueText = (TextView) findViewById(R.id.clipLimitValueText);
		clipLimitValueText.setText(String.valueOf(currentClipLimit));
		clipLimitSeekBar
				.setProgress((int) (currentClipLimit * 100 / DetectorConstants.MAX_CLIP_LIMIT));
		clipLimitSeekBar
				.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
					@Override
					public void onStopTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onStartTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onProgressChanged(SeekBar seekBar,
							int progress, boolean fromUser) {
						int currentValue = ((progress * (DetectorConstants.MAX_CLIP_LIMIT - DetectorConstants.MIN_CLIP_LIMIT)) / 100)
								+ DetectorConstants.MIN_CLIP_LIMIT;
						clipLimitValueText.setText(String.valueOf(currentValue));
						currentClipLimit = currentValue;
					}
				});

		tileSizeSeekBar = (VerticalSeekBar) findViewById(R.id.tilesSizeSeekBar);
		tileSizeValueText = (TextView) findViewById(R.id.tilesSizeValueText);
		tileSizeValueText.setText(String.valueOf(currentTileSize));
		tileSizeSeekBar
				.setProgress((int) (currentTileSize * 100 / DetectorConstants.MAX_TILE_SIZE));
		tileSizeSeekBar
				.setOnSeekBarChangeListener(new OnSeekBarChangeListener() {
					@Override
					public void onStopTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onStartTrackingTouch(SeekBar seekBar) {
					}

					@Override
					public void onProgressChanged(SeekBar seekBar,
							int progress, boolean fromUser) {
						int currentValue = ((progress * (DetectorConstants.MAX_TILE_SIZE - DetectorConstants.MIN_TILE_SIZE)) / 100)
								+ DetectorConstants.MIN_TILE_SIZE;
						tileSizeValueText.setText(String.valueOf(currentValue));
						currentTileSize = currentValue;
					}
				});
	}

	public native void EqualizeHistogram(long matAddrGr);

	public native void FindFace(long matAddrGr, long matAddrRgba);

	public native void FindEyes(long matAddrGr, long matAddrRgba);

	public native void PrepareFindFace(String fileName);

	public native void PrepareFindEyes(String fileName);

	public native void ApplyCLAHE(long matAddrSrc, double clipLimit);

	public native void ApplyCLAHEExt(long matAddrSrc, double clipLimit,
			double tileWidth, double tileHeight);

	public native void SegmentSkin(long matAddrSrc, long matAddrDst);

	public native void FindFeatures(long matAddrGray, long matAddrRgb);

	public native void FindCornerHarris(long matAddrGray, long matAddrDst);

	public native void CalculateOpticalFlow(long matAddrSrc, long matAddrDst);

}
