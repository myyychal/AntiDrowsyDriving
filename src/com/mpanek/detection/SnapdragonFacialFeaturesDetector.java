package com.mpanek.detection;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import android.content.res.Configuration;
import android.util.Log;
import android.view.Display;

import com.qualcomm.snapdragon.sdk.face.FaceData;
import com.qualcomm.snapdragon.sdk.face.FacialProcessing;
import com.qualcomm.snapdragon.sdk.face.FacialProcessing.PREVIEW_ROTATION_ANGLE;

public class SnapdragonFacialFeaturesDetector {

	FacialProcessing faceProc;
	FaceData[] faceArray = null;
	Display display;
	String TAG;
	int mCameraId;

	public SnapdragonFacialFeaturesDetector() {
		super();
		faceProc = FacialProcessing.getInstance();
	}

	public SnapdragonFacialFeaturesDetector(Display display, String tAG, int mCameraId) {
		super();
		this.display = display;
		TAG = tAG;
		this.mCameraId = mCameraId;
	}

	public SnapdragonFacialFeaturesDetector(FacialProcessing faceProc, FaceData[] faceArray, Display display, int mCameraId, String tag) {
		super();
		this.faceProc = faceProc;
		this.faceArray = faceArray;
		this.display = display;
		this.mCameraId = mCameraId;
		this.TAG = tag;
	}

	public FacialProcessing getFaceProc() {
		return faceProc;
	}

	public void setFaceProc(FacialProcessing faceProc) {
		this.faceProc = faceProc;
	}

	public FaceData[] getFaceArray() {
		return faceArray;
	}

	public void setFaceArray(FaceData[] faceArray) {
		this.faceArray = faceArray;
	}

	public Display getDisplay() {
		return display;
	}

	public void setDisplay(Display display) {
		this.display = display;
	}

	public int getmCameraId() {
		return mCameraId;
	}

	public void setmCameraId(int mCameraId) {
		this.mCameraId = mCameraId;
	}

	public String getTAG() {
		return TAG;
	}

	public void setTAG(String tAG) {
		TAG = tAG;
	}

	public Mat findFace(Mat img, int orientation) {
		int dRotation = display.getRotation();
		PREVIEW_ROTATION_ANGLE angleEnum = PREVIEW_ROTATION_ANGLE.ROT_0;
		switch (dRotation) {
		case 0:
			angleEnum = PREVIEW_ROTATION_ANGLE.ROT_90;
			break;

		case 1:
			angleEnum = PREVIEW_ROTATION_ANGLE.ROT_0;
			break;

		case 2:
			// This case is never reached.
			break;

		case 3:
			angleEnum = PREVIEW_ROTATION_ANGLE.ROT_180;
			break;
		}

		if (faceProc == null) {
			faceProc = FacialProcessing.getInstance();
		}

		Mat newMat = new Mat(img.height(), img.width(), CvType.CV_8UC4);

		if (img.channels() >= 3) {
			Imgproc.cvtColor(img, newMat, Imgproc.COLOR_RGB2YUV_I420);
		} else {
			Imgproc.cvtColor(img, newMat, Imgproc.COLOR_GRAY2RGB);
			Imgproc.cvtColor(newMat, newMat, Imgproc.COLOR_RGB2YUV_I420);
		}

		Imgproc.equalizeHist(newMat, newMat);

		boolean isFaceDetected = false;

		byte[] buff = new byte[(int) (newMat.total() * newMat.elemSize())];
		newMat.get(0, 0, buff);

		int width = newMat.width();
		int height = (newMat.height() * 2) / 3;

		Log.i(TAG, "DIMENSIONS: data - " + String.valueOf(buff.length));
		Log.i(TAG, "DIMENSIONS: previewWidth - " + String.valueOf(width));
		Log.i(TAG, "DIMENSIONS: previewHeight - " + String.valueOf(height));

		// Landscape mode - front camera
		if (orientation == Configuration.ORIENTATION_LANDSCAPE && mCameraId == CameraBridgeViewBase.CAMERA_ID_FRONT) {
			isFaceDetected = faceProc.setFrame(buff, width, height, false, angleEnum);
			Log.i(TAG, "DetectFace: landscape front, angleNum: " + angleEnum);
		}
		// Landscape mode - back camera
		else if (orientation == Configuration.ORIENTATION_LANDSCAPE && mCameraId == CameraBridgeViewBase.CAMERA_ID_BACK) {
			isFaceDetected = faceProc.setFrame(buff, width, height, false, angleEnum);
			Log.i(TAG, "DetectFace: landscape back, angleNum: " + angleEnum);
		}

		Log.i(TAG, "isFaceDetected: " + String.valueOf(isFaceDetected));

		faceProc.setProcessingMode(FacialProcessing.FP_MODES.FP_MODE_VIDEO);
		faceArray = faceProc.getFaceData();
		faceProc.normalizeCoordinates(width, height);

		if (faceArray != null) {
			Point leftEye = new Point();
			leftEye.x = faceArray[0].leftEye.x;
			leftEye.y = faceArray[0].leftEye.y;
			Core.circle(img, leftEye, 5, new Scalar(255, 255, 255));

			Point rightEye = new Point();
			rightEye.x = faceArray[0].rightEye.x;
			rightEye.y = faceArray[0].rightEye.y;
			Core.circle(img, rightEye, 5, new Scalar(255, 255, 255));

			Point mouth = new Point();
			mouth.x = faceArray[0].mouth.x;
			mouth.y = faceArray[0].mouth.y;
			Core.circle(img, mouth, 5, new Scalar(255, 255, 255));

			Log.i(TAG, "leftEye: " + String.valueOf(faceArray[0].leftEye.toString()));
			Log.i(TAG, "rightEye: " + String.valueOf(faceArray[0].rightEye.toString()));
			Log.i(TAG, "mouth: " + String.valueOf(faceArray[0].mouth.toString()));
			Log.i(TAG, "face: " + String.valueOf(faceArray[0].rect.toString()));
		} else {
			Core.putText(img, "No face", new Point(newMat.width() / 2, newMat.height() / 2), Core.FONT_HERSHEY_COMPLEX_SMALL, 0.8, new Scalar(200,
					200, 250));
		}

		return img;
	}

}
