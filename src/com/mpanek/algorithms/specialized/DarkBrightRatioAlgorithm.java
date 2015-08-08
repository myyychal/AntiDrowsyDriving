package com.mpanek.algorithms.specialized;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.mpanek.algorithms.general.BinarizationAlgorithm;
import com.mpanek.algorithms.general.ClaheAlgorithm;
import com.mpanek.algorithms.general.HistogramEqualizationAlgorithm;

public class DarkBrightRatioAlgorithm {
	
	private static final String TAG = "AntiDrowsyDriving::DarkBrightRatioAlgorithm";

	ClaheAlgorithm claheAlgorithm;
	HistogramEqualizationAlgorithm histogramEqualizationAlgorithm;
	BinarizationAlgorithm binarizationAlgorithm;

	int erosionSize = 1;

	public DarkBrightRatioAlgorithm() {
		claheAlgorithm = new ClaheAlgorithm();
		histogramEqualizationAlgorithm = new HistogramEqualizationAlgorithm();
		binarizationAlgorithm = new BinarizationAlgorithm();
	}

	public DarkBrightRatioAlgorithm(ClaheAlgorithm claheAlgorithm, HistogramEqualizationAlgorithm histogramEqualizationAlgorithm,
			BinarizationAlgorithm binarizationAlgorithm) {
		this.claheAlgorithm = claheAlgorithm;
		this.histogramEqualizationAlgorithm = histogramEqualizationAlgorithm;
		this.binarizationAlgorithm = binarizationAlgorithm;
	}

	public void simpleEqualizeAndSimpleBinarize(Mat frame) {
		histogramEqualizationAlgorithm.standardEqualizationJava(frame);
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		binarizationAlgorithm.standardBinarization(frame);
	}

	public void simpleEqualizeAndAdaptiveBinarize(Mat frame) {
		histogramEqualizationAlgorithm.standardEqualizationJava(frame);
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		binarizationAlgorithm.adaptiveMeanBinarization(frame);
	}

	public void claheEqualizeAndSimpleBinarize(Mat frame) {
		claheAlgorithm.process(frame);
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		binarizationAlgorithm.standardBinarization(frame);
	}

	public void claheEqualizeAndAdaptiveBinarize(Mat frame) {
		claheAlgorithm.process(frame);
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		binarizationAlgorithm.adaptiveMeanBinarization(frame);
	}

	public void claheEqualizeSimpleBinarizeAndCloseOperation(Mat frame) {
		claheAlgorithm.process(frame);
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		binarizationAlgorithm.standardBinarization(frame);

		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * erosionSize + 1, 2 * erosionSize + 1));
		Imgproc.erode(frame, frame, element);
		Imgproc.dilate(frame, frame, element);
	}
	
	public void claheEqualizeAdaptiveBinarizeAndCloseOperation(Mat frame) {
		claheAlgorithm.process(frame);
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		binarizationAlgorithm.adaptiveMeanBinarization(frame);

		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * erosionSize + 1, 2 * erosionSize + 1));
		Imgproc.erode(frame, frame, element);
		Imgproc.dilate(frame, frame, element);
	}

	public ClaheAlgorithm getClaheAlgorithm() {
		return claheAlgorithm;
	}

	public void setClaheAlgorithm(ClaheAlgorithm claheAlgorithm) {
		this.claheAlgorithm = claheAlgorithm;
	}

	public HistogramEqualizationAlgorithm getHistogramEqualizationAlgorithm() {
		return histogramEqualizationAlgorithm;
	}

	public void setHistogramEqualizationAlgorithm(HistogramEqualizationAlgorithm histogramEqualizationAlgorithm) {
		this.histogramEqualizationAlgorithm = histogramEqualizationAlgorithm;
	}

	public BinarizationAlgorithm getBinarizationAlgorithm() {
		return binarizationAlgorithm;
	}

	public void setBinarizationAlgorithm(BinarizationAlgorithm binarizationAlgorithm) {
		this.binarizationAlgorithm = binarizationAlgorithm;
	}

	public int getErosionSize() {
		return erosionSize;
	}

	public void setErosionSize(int erosionSize) {
		this.erosionSize = erosionSize;
	}

}
