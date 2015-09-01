package com.mpanek.activities.main;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.DialogInterface.OnClickListener;
import android.graphics.Color;
import android.hardware.SensorManager;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.os.Vibrator;
import android.util.Log;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.OrientationEventListener;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnLongClickListener;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.PopupMenu;
import android.widget.PopupMenu.OnMenuItemClickListener;
import android.widget.RelativeLayout;
import android.widget.Toast;

import com.mpanek.algorithms.general.BinarizationAlgorithm;
import com.mpanek.algorithms.general.ClaheAlgorithm;
import com.mpanek.algorithms.general.EdgeDetectionAlgorithm;
import com.mpanek.algorithms.general.HistogramEqualizationAlgorithm;
import com.mpanek.algorithms.specialized.DarkBrightRatioAlgorithm;
import com.mpanek.constants.ViewModesConstants;
import com.mpanek.detection.elements.SnapdragonFacialFeaturesDetector;
import com.mpanek.detection.elements.eyes.CascadeEyesDetector;
import com.mpanek.detection.elements.face.CascadeFaceDetector;
import com.mpanek.detection.elements.face.ColorSegmentationFaceDetector;
import com.mpanek.detection.elements.mouth.CascadeMouthDetector;
import com.mpanek.detection.elements.nose.CascadeNoseDetector;
import com.mpanek.detection.main.DrowsinessDetector;
import com.mpanek.utils.VisualUtils;
import com.mpanek.views.camera.CustomCameraView;

public class NormalActivity extends Activity implements CvCameraViewListener2 {
	private static final String TAG = "AntiDrowsyDriving::NormalActivity";

	private Button startDrowsinessDetectionButton;
	private Button stopAlarmButton;
	
	private ImageView coffeeImage;

	private int mCameraId = CameraBridgeViewBase.CAMERA_ID_FRONT;

	private int mViewMode;

	private Mat mRgba;
	private Mat mGray;
	private Mat currentlyUsedFrame;

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

	private static boolean isCurrentFrameRgb = true;

	private static boolean isGuiHidden = false;

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
				cascadeFaceDetector.prepare(is, cascadeDir);

				int eyesResourceId = getResources().getIdentifier("haarcascade_righteye_2splits", "raw", getPackageName());
				is = getResources().openRawResource(eyesResourceId);
				cascadeEyesDetector.prepare(is, cascadeDir);

				eyesResourceId = getResources().getIdentifier("haarcascade_mcs_lefteye", "raw", getPackageName());
				is = getResources().openRawResource(eyesResourceId);
				cascadeLeftEyeDetector.prepare(is, cascadeDir);

				eyesResourceId = getResources().getIdentifier("haarcascade_mcs_righteye", "raw", getPackageName());
				is = getResources().openRawResource(eyesResourceId);
				cascadeRightEyeDetector.prepare(is, cascadeDir);

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

	public NormalActivity() {
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

	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		if ((keyCode == KeyEvent.KEYCODE_BACK)) {
			finish();
		}
		return super.onKeyDown(keyCode, event);
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.activity_normal);

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
		drowsinessDetector.setVibrator((Vibrator) getSystemService(Context.VIBRATOR_SERVICE));

		coffeeImage = (ImageView)findViewById(R.id.coffeeImage);
		
		stopAlarmButton = (Button) findViewById(R.id.stopAlarmButton);

		startDrowsinessDetectionButton = (Button) findViewById(R.id.startDetectionButton);
		startDrowsinessDetectionButton.setOnLongClickListener(new OnLongClickListener() {
			@Override
			public boolean onLongClick(View v) {
				drowsinessDetector.setAllClosedEyeDetectionMethods(false);
				drowsinessDetector.setSeparateEyesDetection(false);
				drowsinessDetector.setMeanValuesHPA(true);
				drowsinessDetector.setMeanValuesVPA(true);
				AlertDialog dialog;
				AlertDialog.Builder builder = new AlertDialog.Builder(NormalActivity.this);
				builder.setTitle("Select algorithm to detect closed eyes");
				CharSequence[] items = new CharSequence[] { "Mean HPA", "Mean VPA" };
				boolean[] checkedItems = new boolean[] { true, true };
				builder.setMultiChoiceItems(items, checkedItems, new Dialog.OnMultiChoiceClickListener() {
					@Override
					public void onClick(DialogInterface dialog, int indexSelected, boolean isChecked) {
						if (indexSelected == 0) {
							drowsinessDetector.setMeanValuesHPA(isChecked);
						} else {
							drowsinessDetector.setMeanValuesVPA(isChecked);
						}
					}
				});
				builder.setPositiveButton("Ok", new OnClickListener() {
					@Override
					public void onClick(DialogInterface arg0, int arg1) {

						AlertDialog dialog;
						AlertDialog.Builder builder = new AlertDialog.Builder(NormalActivity.this);
						builder.setTitle("Select way to detect drowsiness");
						CharSequence[] items = new CharSequence[] { "Long eye time closed", "PERCLOSE" };
						boolean[] checkedItems = new boolean[] { true, true };
						builder.setMultiChoiceItems(items, checkedItems, new Dialog.OnMultiChoiceClickListener() {
							@Override
							public void onClick(DialogInterface dialog, int indexSelected, boolean isChecked) {
								if (indexSelected == 0) {
									drowsinessDetector.setFirstMethod(isChecked);
								} else {
									drowsinessDetector.setSecondMethod(isChecked);
								}
							}
						});
						builder.setPositiveButton("Ok", new OnClickListener() {

							@Override
							public void onClick(DialogInterface dialoginterface, int i) {
								LayoutInflater li = LayoutInflater.from(getBaseContext());
								View promptsView = li.inflate(R.layout.drowiness_options, null);

								final EditText maxEyeClosedTimeEditText = (EditText) promptsView.findViewById(R.id.maxEyeClosedTimeEditText);
								final EditText warningsCounterEditText = (EditText) promptsView.findViewById(R.id.warningsCounterEditText);
								final EditText warningCounterReduceTimeEditText = (EditText) promptsView
										.findViewById(R.id.warningCounterReduceTimeEditText);
								//final EditText perclosTimeEditText = (EditText) promptsView.findViewById(R.id.perclosTimeEditText);

								maxEyeClosedTimeEditText.setText(String.valueOf(drowsinessDetector.getTimeForFirstMethod()));
								warningsCounterEditText.setText(String.valueOf(drowsinessDetector.getMaxFirstMethodAlertCounter()));
								warningCounterReduceTimeEditText.setText(String.valueOf(drowsinessDetector.getTimeToReduceAlertCounter()));
								//perclosTimeEditText.setText(String.valueOf(drowsinessDetector.getTimeForSecondMethod()));

								AlertDialog dialog;
								AlertDialog.Builder builder = new AlertDialog.Builder(NormalActivity.this);
								builder.setTitle("Parameters");
								builder.setView(promptsView);
								builder.setPositiveButton("Ok", new OnClickListener() {
									@Override
									public void onClick(DialogInterface dialog, int which) {
										drowsinessDetector.setTimeForFirstMethod(Long.parseLong(maxEyeClosedTimeEditText.getText().toString()));
										drowsinessDetector.setMaxFirstMethodAlertCounter(Integer.parseInt(warningsCounterEditText.getText()
												.toString()));
										drowsinessDetector.setTimeToReduceAlertCounter(Long.parseLong(warningCounterReduceTimeEditText.getText()
												.toString()));
										//drowsinessDetector.setTimeForSecondMethod(Long.parseLong(perclosTimeEditText.getText().toString()));
										drowsinessDetector.setStartDrowsinessDetectionTime(System.currentTimeMillis());
										mViewMode = ViewModesConstants.VIEW_MODE_FINAL_SOLUTION;
									}
								});
								dialog = builder.create();
								dialog.show();

							}
						});
						dialog = builder.create();
						dialog.show();

					}
				});
				dialog = builder.create();
				dialog.show();
				return true;
			}
		});

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
		Log.i(TAG, "onDestroy");
		super.onPause();
		if (mCustomCameraView != null)
			mCustomCameraView.disableView();
	}

	@Override
	public void onResume() {
		Log.i(TAG, "onDestroy");
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
	}

	public void onDestroy() {
		Log.i(TAG, "onDestroy");
		super.onDestroy();
		if (mCustomCameraView != null) {
			mCustomCameraView.disableView();
		}
	}

	public void onCameraViewStarted(int width, int height) {
		mRgba = new Mat(height, width, CvType.CV_8UC4);
		currentlyUsedFrame = new Mat();
		mGray = new Mat(height, width, CvType.CV_8UC1);
	}

	public void onCameraViewStopped() {
		mRgba.release();
		mGray.release();
		currentlyUsedFrame.release();

	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();

		if (stopAlarmButton.getVisibility() != View.VISIBLE && drowsinessDetector.isMainAlarmOn()) {
			runOnUiThread(new Runnable() {
				@Override
				public void run() {
					stopAlarmButton.setVisibility(View.VISIBLE);
				}
			});
		}
		if (drowsinessDetector.isCoffeeNeeded() && coffeeImage.getVisibility() != View.VISIBLE){
			runOnUiThread(new Runnable() {
				@Override
				public void run() {
					coffeeImage.setVisibility(View.VISIBLE);
				}
			});
		} else if (!drowsinessDetector.isCoffeeNeeded() && coffeeImage.getVisibility() != View.INVISIBLE) {
			runOnUiThread(new Runnable() {
				@Override
				public void run() {
					coffeeImage.setVisibility(View.INVISIBLE);
				}
			});
		}

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

		switch (viewMode) {

		case ViewModesConstants.VIEW_MODE_NONE:
			break;

		case ViewModesConstants.VIEW_MODE_START_DROWSINESS_DETECTION:
			currentlyUsedFrame = drowsinessDetector.processDetection(mGray, mRgba);
			break;

		case ViewModesConstants.VIEW_MODE_FINAL_SOLUTION:
			currentlyUsedFrame = drowsinessDetector.processDetectionFinal(mGray, mRgba);
			return mRgba;

		}

		return currentlyUsedFrame;
	}

	public void showOptionsMenu(View v) {
		if (mCustomCameraView != null) {
			Log.i(TAG, ((CustomCameraView) mCustomCameraView).getCameraInfo());
		}
		openOptionsMenu();
	}

	public void stopAlarm(View v) {
		drowsinessDetector.setMainAlarmOn(false);
		drowsinessDetector.getToneGenerator().startTone(ToneGenerator.TONE_PROP_BEEP);
		runOnUiThread(new Runnable() {
			@Override
			public void run() {
				stopAlarmButton.setVisibility(View.INVISIBLE);
			}
		});
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

	public void takePicture(View v) {
		Button pictureTakingButton = (Button) findViewById(R.id.takePictureButton);
		if (VisualUtils.isPictureTakingAllowed) {
			VisualUtils.isPictureTakingAllowed = false;
			pictureTakingButton.setBackgroundColor(Color.GREEN);
		} else {
			VisualUtils.isPictureTakingAllowed = true;
			pictureTakingButton.setBackgroundColor(Color.RED);
		}
	}

	public void resetView(View v) {
		mViewMode = ViewModesConstants.VIEW_MODE_NONE;
		runOnUiThread(new Runnable() {
			@Override
			public void run() {
				coffeeImage.setVisibility(View.INVISIBLE);
			}
		});
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
				AlertDialog.Builder builder = new AlertDialog.Builder(NormalActivity.this);
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
						AlertDialog.Builder builder = new AlertDialog.Builder(NormalActivity.this);
						builder.setTitle("Select eye openess algorithm");
						CharSequence[] algorithms = new CharSequence[] { "DarkBright - adaptive binarization", "DarkBright - simple binarization",
								"Edge detection - Laplacian", "Mean values vertical projection analysis", "Intensity vertical projection analysis",
								"Mean values horizontal projection analysis", "Intensity horizontal projection analysis", "Do nothing" };
						drowsinessDetector.setAllClosedEyeDetectionMethods(false);
						drowsinessDetector.setAdaptiveBinarizationUsed(true);
						builder.setSingleChoiceItems(algorithms, 0, new OnClickListener() {
							@Override
							public void onClick(DialogInterface arg0, int arg1) {
								if (arg1 == 0) {
									drowsinessDetector.setAllClosedEyeDetectionMethods(false);
									drowsinessDetector.setAdaptiveBinarizationUsed(true);
								} else if (arg1 == 1) {
									drowsinessDetector.setAllClosedEyeDetectionMethods(false);
									drowsinessDetector.setSimpleBinarizationUsed(true);
								} else if (arg1 == 2) {
									drowsinessDetector.setAllClosedEyeDetectionMethods(false);
									drowsinessDetector.setLaplacianAlgorithmUsed(true);
								} else if (arg1 == 3) {
									drowsinessDetector.setAllClosedEyeDetectionMethods(false);
									drowsinessDetector.setProjectionAnalysis(true);
									drowsinessDetector.setMeanValuesVPA(true);
								} else if (arg1 == 4) {
									drowsinessDetector.setAllClosedEyeDetectionMethods(false);
									drowsinessDetector.setProjectionAnalysis(true);
									drowsinessDetector.setIntensityVPA(true);
								} else if (arg1 == 5) {
									drowsinessDetector.setAllClosedEyeDetectionMethods(false);
									drowsinessDetector.setProjectionAnalysis(true);
									drowsinessDetector.setMeanValuesHPA(true);
								} else if (arg1 == 6) {
									drowsinessDetector.setAllClosedEyeDetectionMethods(false);
									drowsinessDetector.setProjectionAnalysis(true);
									drowsinessDetector.setIntensityHPA(true);
								} else if (arg1 == 7) {
									drowsinessDetector.setAllClosedEyeDetectionMethods(false);
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
			if (!(parentView instanceof CustomCameraView 
					|| parentView.getId() == R.id.showHideGui 
					|| parentView.getId() == R.id.startDetectionButton
					|| parentView.getId() == R.id.stopAlarmButton)) {
				parentView.setVisibility(visibiltyFlag);
			}
		}
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

}
