package com.mpanek.detection;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import android.util.Log;

public class CascadeDetector {

	protected String cascadeFileName;
	protected File cascadeFile;
	protected CascadeClassifier javaDetector;

	protected String TAG;

	protected boolean isSizeManuallyChanged = false;

	protected float scaleFactor = 1.1f;
	protected int minNeighbours = 2;
	protected int detectionFlag = Objdetect.CASCADE_SCALE_IMAGE;

	protected float mRelativeMinObjectSize;
	protected int mAbsoluteMinObjectSize;
	protected float mRelativeMaxObjectSize;
	protected int mAbsoluteMaxObjectSize;

	public CascadeDetector() {
		super();
	}

	public CascadeDetector(String cascadeFileName, File cascadeFile, CascadeClassifier javaDetector) {
		super();
		this.cascadeFileName = cascadeFileName;
		this.cascadeFile = cascadeFile;
		this.javaDetector = javaDetector;
	}

	public String getCascadeFileName() {
		return cascadeFileName;
	}

	public void setCascadeFileName(String cascadeFileName) {
		this.cascadeFileName = cascadeFileName;
	}

	public File getCascadeFile() {
		return cascadeFile;
	}

	public void setCascadeFile(File cascadeFile) {
		this.cascadeFile = cascadeFile;
	}

	public CascadeClassifier getJavaDetector() {
		return javaDetector;
	}

	public void setJavaDetector(CascadeClassifier javaDetector) {
		this.javaDetector = javaDetector;
	}

	public boolean isSizeManuallyChanged() {
		return isSizeManuallyChanged;
	}

	public void setSizeManuallyChanged(boolean isSizeManuallyChanged) {
		this.isSizeManuallyChanged = isSizeManuallyChanged;
	}

	public float getScaleFactor() {
		return scaleFactor;
	}

	public void setScaleFactor(float scaleFactor) {
		this.scaleFactor = scaleFactor;
	}

	public int getMinNeighbours() {
		return minNeighbours;
	}

	public void setMinNeighbours(int minNeighbours) {
		this.minNeighbours = minNeighbours;
	}

	public int getDetectionFlag() {
		return detectionFlag;
	}

	public void setDetectionFlag(int detectionFlag) {
		this.detectionFlag = detectionFlag;
	}

	public float getmRelativeMinObjectSize() {
		return mRelativeMinObjectSize;
	}

	public void setmRelativeMinObjectSize(float mRelativeMinObjectSize) {
		this.mRelativeMinObjectSize = mRelativeMinObjectSize;
	}

	public int getmAbsoluteMinObjectSize() {
		return mAbsoluteMinObjectSize;
	}

	public void setmAbsoluteMinObjectSize(int mAbsoluteMinObjectSize) {
		this.mAbsoluteMinObjectSize = mAbsoluteMinObjectSize;
	}

	public float getmRelativeMaxObjectSize() {
		return mRelativeMaxObjectSize;
	}

	public void setmRelativeMaxObjectSize(float mRelativeMaxObjectSize) {
		this.mRelativeMaxObjectSize = mRelativeMaxObjectSize;
	}

	public int getmAbsoluteMaxObjectSize() {
		return mAbsoluteMaxObjectSize;
	}

	public void setmAbsoluteMaxObjectSize(int mAbsoluteMaxObjectSize) {
		this.mAbsoluteMaxObjectSize = mAbsoluteMaxObjectSize;
	}

	public File prepare(InputStream is, File cascadeDir) {
		try {

			cascadeFile = new File(cascadeDir, cascadeFileName);

			FileOutputStream os = new FileOutputStream(cascadeFile);

			byte[] buffer = new byte[4096];
			int bytesRead;
			while ((bytesRead = is.read(buffer)) != -1) {
				os.write(buffer, 0, bytesRead);
			}
			is.close();
			os.close();

			javaDetector = new CascadeClassifier(cascadeFile.getAbsolutePath());
			if (javaDetector.empty()) {
				Log.e(TAG, "Failed to load cascade classifier");
				javaDetector = null;
			} else
				Log.i(TAG, "Loaded cascade classifier from " + cascadeFile.getAbsolutePath());

			cascadeDir.delete();

		} catch (IOException e) {
			e.printStackTrace();
			Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
		}
		return cascadeFile;
	}

}
