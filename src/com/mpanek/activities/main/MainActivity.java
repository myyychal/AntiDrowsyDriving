package com.mpanek.activities.main;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

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
import android.hardware.SensorManager;
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
import android.view.WindowManager;
import android.widget.PopupMenu;
import android.widget.PopupMenu.OnMenuItemClickListener;
import android.widget.SeekBar;
import android.widget.SeekBar.OnSeekBarChangeListener;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VerticalSeekBar;

import com.mpanek.constants.DetectorConstants;
import com.mpanek.constants.DrawingConstants;
import com.mpanek.detection.SnapdragonFacialFeaturesDetector;
import com.mpanek.detection.eyes.CascadeEyesDetector;
import com.mpanek.detection.face.CascadeFaceDetector;
import com.mpanek.detection.face.ColorSegmentationFaceDetector;
import com.mpanek.detection.mouth.CascadeMouthDetector;
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

	private int mCameraId = CameraBridgeViewBase.CAMERA_ID_FRONT;

	private static final int VIEW_MODE_NONE = 0;
	private static final int VIEW_MODE_EQ_NONE = 3;
	private static final int VIEW_MODE_EQ_HIST_CPP = 5;
	private static final int VIEW_MODE_EQ_HIST_CLAHE_CPP = 4;
	private static final int VIEW_MODE_FIND_FACE_SNAPDRAGON = 6;
	private static final int VIEW_MODE_FIND_FACE_CASCADE_JAVA = 7;
	private static final int VIEW_MODE_FIND_FACE_CASCADE_CPP = 8;
	private static final int VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_JAVA = 12;
	private static final int VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_CPP = 13;
	private static final int VIEW_MODE_FIND_EYES_CASCADE_JAVA = 9;
	private static final int VIEW_MODE_FIND_EYES_CASCADE_CPP = 10;
	private static final int VIEW_MODE_FIND_MOUTH_CASCADE_JAVA = 11;
	private static final int VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_JAVA = 14;
	private static final int VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_CPP = 15;
	private static final int VIEW_MODE_FIND_CORNER_HARRIS_JAVA = 16;
	private static final int VIEW_MODE_FIND_CORNER_HARRIS_CPP = 18;
	private static final int VIEW_MODE_OPTICAL_FLOW_JAVA = 17;
	private static final int VIEW_MODE_OPTICAL_FLOW_CPP = 19;
	private static final int VIEW_MODE_START_DROWSINESS_DETECTION = 20;

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
	ColorSegmentationFaceDetector colorSegmentationFaceDetector;

	OrientationEventListener orientationEventListener;
	int deviceOrientation;

	private static boolean isMouthWithEyes = true;
	private static boolean isFaceTracking = true;
	private static boolean isCurrentFrameRgb = true;

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

		colorSegmentationFaceDetector = new ColorSegmentationFaceDetector();

		VerticalSeekBar scaleFactorSeekBar = (VerticalSeekBar) findViewById(R.id.scaleFactorSeekBar);
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
						case VIEW_MODE_FIND_FACE_CASCADE_JAVA:
							cascadeFaceDetector.setScaleFactor(currentValue);
							break;

						case VIEW_MODE_FIND_EYES_CASCADE_JAVA:
							cascadeEyesDetector.setScaleFactor(currentValue);
							break;
						}
					}
				});

		VerticalSeekBar minNeighsSeekBar = (VerticalSeekBar) findViewById(R.id.minNeighsSeekBar);
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
						case VIEW_MODE_FIND_FACE_CASCADE_JAVA:
							cascadeFaceDetector.setMinNeighbours(currentValue);
							break;

						case VIEW_MODE_FIND_EYES_CASCADE_JAVA:
							cascadeEyesDetector.setMinNeighbours(currentValue);
							break;
						}
					}
				});

		VerticalSeekBar minFaceSeekBar = (VerticalSeekBar) findViewById(R.id.minFaceSeekBar);
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

		VerticalSeekBar maxFaceSeekBar = (VerticalSeekBar) findViewById(R.id.maxFaceSeekBar);
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

		VerticalSeekBar minEyeSeekBar = (VerticalSeekBar) findViewById(R.id.minEyeSeekBar);
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

		VerticalSeekBar maxEyeSeekBar = (VerticalSeekBar) findViewById(R.id.maxEyeSeekBar);
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

		VerticalSeekBar clipLimitSeekBar = (VerticalSeekBar) findViewById(R.id.clipLimitSeekBar);
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

		VerticalSeekBar tileSizeValueBar = (VerticalSeekBar) findViewById(R.id.tilesSizeSeekBar);
		tileSizeValueText = (TextView) findViewById(R.id.tilesSizeValueText);
		tileSizeValueText.setText(String.valueOf(currentTileSize));
		tileSizeValueBar
				.setProgress((int) (currentTileSize * 100 / DetectorConstants.MAX_TILE_SIZE));
		tileSizeValueBar
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
			Core.flip(inputFrame.rgba(), mRgba, 1);
			Core.flip(inputFrame.gray(), mGray, 1);
		}

		if (isCurrentFrameRgb) {
			currentlyUsedFrame = mRgba;
		} else {
			currentlyUsedFrame = mGray;
		}

		final int viewMode = mViewMode;
		final int eqHistMode = mEqHistMode;

		switch (eqHistMode) {
		case VIEW_MODE_EQ_NONE:
			break;

		case VIEW_MODE_EQ_HIST_CPP:
			EqualizeHistogram(currentlyUsedFrame.getNativeObjAddr());
			break;

		case VIEW_MODE_EQ_HIST_CLAHE_CPP:
			ApplyCLAHEExt(currentlyUsedFrame.getNativeObjAddr(),
					(double) currentClipLimit, (double) currentTileSize,
					(double) currentTileSize);
			break;
		}

		switch (viewMode) {
		
		case VIEW_MODE_NONE:
			break;
		
		case VIEW_MODE_FIND_FACE_SNAPDRAGON:
			snapdragonFacialFeaturesDetector.findFace(mRgba, this
					.getResources().getConfiguration().orientation);
			break;

		case VIEW_MODE_FIND_FACE_CASCADE_JAVA:
			Rect face = cascadeFaceDetector.findFace(currentlyUsedFrame);
			if (face != null) {
				VisualUtils.drawRect(face, currentlyUsedFrame,
						DrawingConstants.FACE_RECT_COLOR);
			}
			break;

		case VIEW_MODE_FIND_FACE_CASCADE_CPP:
			FindFace(currentlyUsedFrame.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_JAVA:
			currentlyUsedFrame = colorSegmentationFaceDetector
					.detectFaceYCrCb(currentlyUsedFrame);
			break;

		case VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_CPP:
			SegmentSkin(mRgba.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case VIEW_MODE_FIND_EYES_CASCADE_JAVA:
			Rect foundFace;
			if (isFaceTracking) {
				foundFace = cascadeFaceDetector.findFace(mGray);
			} else {
				foundFace = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFace != null) {
				VisualUtils.drawRect(foundFace, currentlyUsedFrame,
						DrawingConstants.FACE_RECT_COLOR);
				Rect[] eyes = cascadeEyesDetector.findEyes(mGray, foundFace);
				VisualUtils.drawRects(eyes, currentlyUsedFrame,
						DrawingConstants.EYES_RECT_COLOR);
			}
			break;

		case VIEW_MODE_FIND_EYES_CASCADE_CPP:
			FindEyes(currentlyUsedFrame.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case VIEW_MODE_FIND_MOUTH_CASCADE_JAVA:
			Rect foundFace2;
			if (isFaceTracking) {
				foundFace2 = cascadeFaceDetector.findFace(currentlyUsedFrame);
			} else {
				foundFace2 = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFace2 != null) {
				VisualUtils.drawRect(foundFace2, currentlyUsedFrame,
						DrawingConstants.FACE_RECT_COLOR);
				if (isMouthWithEyes) {
					Rect[] eyes = cascadeEyesDetector.findEyes(
							currentlyUsedFrame, foundFace2);
					VisualUtils.drawRects(eyes, currentlyUsedFrame,
							DrawingConstants.EYES_RECT_COLOR);
				}
				Rect[] mouths = cascadeMouthDetector.findMouth(
						currentlyUsedFrame, foundFace2);
				VisualUtils.drawRects(mouths, currentlyUsedFrame,
						DrawingConstants.MOUTH_RECT_COLOR);
			}
			break;

		case VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_JAVA:
			MatOfPoint cornersPoints = new MatOfPoint();
			Imgproc.goodFeaturesToTrack(mGray, cornersPoints, 100, 0.01, 10);
			int rows = cornersPoints.rows();
			List<Point> listOfPoints = cornersPoints.toList();
			for (int x = 0; x < rows; x++) {
				Core.circle(currentlyUsedFrame, listOfPoints.get(x), 5,
						DrawingConstants.RED);
			}
			break;

		case VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_CPP:
			FindFeatures(mGray.getNativeObjAddr(),
					currentlyUsedFrame.getNativeObjAddr());
			break;

		case VIEW_MODE_FIND_CORNER_HARRIS_JAVA:
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
			
		case VIEW_MODE_FIND_CORNER_HARRIS_CPP:
			FindCornerHarris(mGray.getNativeObjAddr(), currentlyUsedFrame.getNativeObjAddr());
			break;

		case VIEW_MODE_OPTICAL_FLOW_JAVA:
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
			
		case VIEW_MODE_OPTICAL_FLOW_CPP:
			CalculateOpticalFlow(mRgba.getNativeObjAddr(), currentlyUsedFrame.getNativeObjAddr());
			break;
			
		case VIEW_MODE_START_DROWSINESS_DETECTION:
			currentlyUsedFrame = mGray;
			EqualizeHistogram(mGray.getNativeObjAddr());
			ApplyCLAHEExt(mGray.getNativeObjAddr(),
					(double) currentClipLimit, (double) currentTileSize,
					(double) currentTileSize);
			Rect foundFaceInDetection;
			if (isFaceTracking) {
				foundFaceInDetection = cascadeFaceDetector.findFace(mGray);
			} else {
				foundFaceInDetection = cascadeFaceDetector.getLastFoundFace();
			}
			if (foundFaceInDetection != null) {
				foundFaceInDetection.height = foundFaceInDetection.height/2;
				foundFaceInDetection.y = (int) (foundFaceInDetection.y + 0.1 * mGray.height()); 
				VisualUtils.drawRect(foundFaceInDetection, currentlyUsedFrame,
						DrawingConstants.FACE_RECT_COLOR);
				Rect[] eyes = cascadeEyesDetector.findEyes(mGray, foundFaceInDetection);
				VisualUtils.drawRects(eyes, currentlyUsedFrame,
						DrawingConstants.EYES_RECT_COLOR);
			}
			break;
			
		}

		return currentlyUsedFrame;
	}

	public boolean onOptionsItemSelected(MenuItem item) {
		if (item.getGroupId() == 1) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = VIEW_MODE_FIND_FACE_SNAPDRAGON;
				break;
			case 1:
				mViewMode = VIEW_MODE_FIND_FACE_CASCADE_JAVA;
				break;
			case 2:
				mViewMode = VIEW_MODE_FIND_FACE_CASCADE_CPP;
				break;
			case 3:
				mViewMode = VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_JAVA;
				break;
			case 4:
				mViewMode = VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_CPP;
				break;
			}
		} else if (item.getGroupId() == 2) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = VIEW_MODE_FIND_EYES_CASCADE_JAVA;
				break;
			case 1:
				mViewMode = VIEW_MODE_FIND_EYES_CASCADE_CPP;
				break;
			}
		} else if (item.getGroupId() == 3) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = VIEW_MODE_FIND_MOUTH_CASCADE_JAVA;
				isMouthWithEyes = false;
				break;
			case 1:
				mViewMode = VIEW_MODE_FIND_MOUTH_CASCADE_JAVA;
				isMouthWithEyes = true;
				break;
			}
		} else if (item.getGroupId() == 4) {
			switch (item.getItemId()) {
			case 0:
				mEqHistMode = VIEW_MODE_EQ_NONE;
				break;
			case 1:
				mEqHistMode = VIEW_MODE_EQ_HIST_CPP;
				break;
			case 2:
				mEqHistMode = VIEW_MODE_EQ_HIST_CLAHE_CPP;
				break;
			}
		} else if (item.getGroupId() == 5) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_JAVA;
				break;
			case 1:
				mViewMode = VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_CPP;
				break;
			case 2:
				mViewMode = VIEW_MODE_FIND_CORNER_HARRIS_JAVA;
				break;
			case 3:
				mViewMode = VIEW_MODE_FIND_CORNER_HARRIS_CPP;
				break;
			case 4:
				mViewMode = VIEW_MODE_OPTICAL_FLOW_JAVA;
				break;
			case 5:
				mViewMode = VIEW_MODE_OPTICAL_FLOW_CPP;
				break;
			}
		}
		return true;
	}
	
	public void resetView(View v){
		mViewMode = VIEW_MODE_NONE;
		mEqHistMode = VIEW_MODE_EQ_NONE;
	}
	
	public void startDrowsinessDetection(View v){
		switch (mViewMode){
		case VIEW_MODE_START_DROWSINESS_DETECTION:
			mViewMode = VIEW_MODE_NONE;
			break;
		case VIEW_MODE_NONE:
			mViewMode = VIEW_MODE_START_DROWSINESS_DETECTION; 
			break;
		}
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