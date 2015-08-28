package com.mpanek.views.camera;

import java.io.FileOutputStream;
import java.util.List;

import org.opencv.android.JavaCameraView;

import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.Size;
import android.util.AttributeSet;
import android.util.Log;

public class CustomCameraView extends JavaCameraView implements PictureCallback {

	private static final String TAG = "AntiDrowsyDriving::CustomCameraView";
	private String mPictureFileName;

	public CustomCameraView(Context context, AttributeSet attrs) {
		super(context, attrs);
	}

	public List<String> getEffectList() {
		return mCamera.getParameters().getSupportedColorEffects();
	}

	public boolean isEffectSupported() {
		return (mCamera.getParameters().getColorEffect() != null);
	}

	public String getEffect() {
		return mCamera.getParameters().getColorEffect();
	}

	public void setEffect(String effect) {
		Camera.Parameters params = mCamera.getParameters();
		params.setColorEffect(effect);
		mCamera.setParameters(params);
	}

	public List<Size> getResolutionList() {
		return mCamera.getParameters().getSupportedPreviewSizes();
	}

	public void setResolution(Size resolution) {
		disconnectCamera();
		mMaxHeight = resolution.height;
		mMaxWidth = resolution.width;
		connectCamera(getWidth(), getHeight());
	}

	public Size getResolution() {
		return mCamera.getParameters().getPreviewSize();
	}

	public void setPreviewFormat(int format) {
		Camera.Parameters params = mCamera.getParameters();
		params.setPreviewFormat(format);
		mCamera.setParameters(params);
	}

	public int getPreviewFormat() {
		return mCamera.getParameters().getPreviewFormat();
	}

	public void setDisplayRotation(int rotation) {
		Camera.Parameters params = mCamera.getParameters();
		params.setRotation(rotation);
		mCamera.setParameters(params);
	}

	public Camera getCamera() {
		return mCamera;
	}

	public String getCameraInfo() {
		return mCamera.getParameters().flatten();
	}

	public void takePicture(final String fileName) {
		Log.i(TAG, "Taking picture");
		this.mPictureFileName = fileName;
		mCamera.setPreviewCallback(null);
		mCamera.takePicture(null, null, this);
	}

	@Override
	public void onPictureTaken(byte[] data, Camera camera) {
		Log.i(TAG, "Saving a bitmap to file");
		mCamera.startPreview();
		mCamera.setPreviewCallback(this);
		try {
			FileOutputStream fos = new FileOutputStream(mPictureFileName);
			fos.write(data);
			fos.close();
		} catch (java.io.IOException e) {
			Log.e("PictureDemo", "Exception in photoCallback", e);
		}

	}

}