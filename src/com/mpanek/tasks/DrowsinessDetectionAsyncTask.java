package com.mpanek.tasks;

import org.opencv.core.Mat;

import com.mpanek.detection.main.DrowsinessDetector;

import android.os.AsyncTask;

public class DrowsinessDetectionAsyncTask extends AsyncTask<Void, Void, Void> {

	Mat mGray, mRgba;
	DrowsinessDetector drowsinessDetector;

	public DrowsinessDetectionAsyncTask(Mat mGray, Mat mRgba, DrowsinessDetector drowsinessDetector) {
		super();
		this.mGray = mGray;
		this.mRgba = mRgba;
		this.drowsinessDetector = drowsinessDetector;
	}

	@Override
	protected Void doInBackground(Void... arg0) {
		drowsinessDetector.processDetection(mGray, mRgba);
		return null;
	}

	public void setGray(Mat mGray) {
		this.mGray = mGray;
	}

	public void setRgba(Mat mRgba) {
		this.mRgba = mRgba;
	}

	public void setDrowsinessDetector(DrowsinessDetector drowsinessDetector) {
		this.drowsinessDetector = drowsinessDetector;
	}

}
