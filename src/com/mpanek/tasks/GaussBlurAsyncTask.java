package com.mpanek.tasks;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.os.AsyncTask;
public class GaussBlurAsyncTask extends AsyncTask<Void, Void, Mat> {

	private static final String TAG = "AntiDrowsyDriving::GaussBlurAsyncTask";

	private Mat frame;
	private int size;

	public GaussBlurAsyncTask(Mat frame, int size) {
		super();
		this.frame = frame;
		this.size = size;
	}

	@Override
	protected Mat doInBackground(Void... params) {
		Imgproc.GaussianBlur(frame, frame, new Size(size, size), 0);
		return frame;
	}

	public void setFrame(Mat frame) {
		this.frame = frame;
	}

	public void setSize(int size) {
		this.size = size;
	}

}
