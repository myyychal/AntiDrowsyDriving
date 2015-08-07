package com.mpanek.algorithms.specialized;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import com.mpanek.algorithms.general.ClaheAlgorithm;
import com.mpanek.algorithms.general.HistogramEqualizationAlgorithm;

public class DarkBrightRatioAlgorithm {
	
	ClaheAlgorithm claheAlgorithm;
	HistogramEqualizationAlgorithm histogramEqualizationAlgorithm;
	
	public DarkBrightRatioAlgorithm(){
		claheAlgorithm = new ClaheAlgorithm();
		histogramEqualizationAlgorithm = new HistogramEqualizationAlgorithm();
	}
	
	public void simpleEqualizeAndBinarize(Mat frame){
		histogramEqualizationAlgorithm.standardEqualizationJava(frame);
		Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		Imgproc.threshold(frame, frame, 120 , 255, Imgproc.THRESH_BINARY);
	}

}
