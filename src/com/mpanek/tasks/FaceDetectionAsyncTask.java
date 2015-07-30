package com.mpanek.tasks;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import com.mpanek.detection.face.CascadeFaceDetector;

import android.os.AsyncTask;

public class FaceDetectionAsyncTask extends AsyncTask<Void, Void, Rect> {

	Mat frame;
	CascadeFaceDetector cascadeFaceDetector;

	public FaceDetectionAsyncTask(Mat frame,
			CascadeFaceDetector cascadeFaceDetector) {
		super();
		this.frame = frame;
		this.cascadeFaceDetector = cascadeFaceDetector;
	}

	@Override
	protected Rect doInBackground(Void... params) {
		if (frame != null) {
			Rect foundFaceInDetection = new Rect(0, 0, frame.width(),
					frame.height());
			Rect boundingBox = new Rect(0, 0, frame.width(), frame.height());
			double boundingMultiplier = 0.1;
			boundingBox.x += boundingMultiplier * frame.width();
			boundingBox.width -= 2 * boundingMultiplier * frame.width();
			foundFaceInDetection = cascadeFaceDetector.findFace(frame,
					boundingBox);
			if (foundFaceInDetection == null) {
				foundFaceInDetection = cascadeFaceDetector.getLastFoundFace();
			}
			return foundFaceInDetection;
		} else
			return null;
	}

	public void setFrame(Mat frame) {
		this.frame = frame;
	}

	public void setCascadeFaceDetector(CascadeFaceDetector cascadeFaceDetector) {
		this.cascadeFaceDetector = cascadeFaceDetector;
	}

}
