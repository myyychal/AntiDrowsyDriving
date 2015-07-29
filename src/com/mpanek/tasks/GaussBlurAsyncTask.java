package com.mpanek.tasks;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.mpanek.constants.DrawingConstants;
import com.mpanek.utils.DrawingUtils;

import android.os.AsyncTask;
import android.util.Log;

public class GaussBlurAsyncTask extends AsyncTask<Void, Void, Mat>{
	
	private static final String TAG = "AntiDrowsyDriving::GaussBlurAsyncTask";
	
	private Mat frame;
	private int size;
	private boolean isFinished = true;

	public GaussBlurAsyncTask(Mat frame, int size) {
		super();
		this.frame = frame;
		this.size = size;
	}

	@Override
	protected Mat doInBackground(Void... params) {
		Log.i(TAG, "gauss: doInBackground, size: " + size);
		Imgproc.GaussianBlur(frame, frame,
				new Size(size, size), 0);
		DrawingUtils.drawRect(new Rect(0, 0, 100, 100), frame, DrawingConstants.GREEN);
		return frame;
	}

	@Override
	protected void onPostExecute(Mat result) {
		Log.i(TAG, "gauss: onPostExecute");
		super.onPostExecute(result);
		isFinished = true;
	}

	@Override
	protected void onPreExecute() {
		Log.i(TAG, "gauss: onPreExecute");
		super.onPreExecute();
		isFinished = false;
	}

	public boolean isFinished() {
		return isFinished;
	}

	public void setFrame(Mat frame) {
		this.frame = frame;
	}

	public void setSize(int size) {
		this.size = size;
	}
	
}
