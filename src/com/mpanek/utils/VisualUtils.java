package com.mpanek.utils;

import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class VisualUtils {
	
	public static void drawRect(Rect rect, Mat imgToDraw, Scalar color, int thickness){
		Core.rectangle(imgToDraw, rect.tl(), rect.br(),
				color, thickness);
	}
	
	public static void drawRect(Rect rect, Mat imgToDraw, Scalar color){
		Core.rectangle(imgToDraw, rect.tl(), rect.br(),
				color, 3);
	}
	
	public static void drawRects(Rect[] rects, Mat imgToDraw, Scalar color, int thickness){
		for (Rect rect : rects){
			drawRect(rect, imgToDraw, color, thickness);
		}
	}
	
	public static void drawRects(Rect[] rects, Mat imgToDraw, Scalar color){
		for (Rect rect : rects){
			drawRect(rect, imgToDraw, color);
		}
	}
	
	public static Mat equalizeIntensity(Mat inputImage)
	{
	    if(inputImage.channels() >= 3)
	    {
	        Mat ycrcb = new Mat();

	        Imgproc.cvtColor(inputImage,ycrcb,Imgproc.COLOR_RGB2YCrCb);

	        ArrayList<Mat> channels = new ArrayList<Mat>();
	        Core.split(ycrcb,channels);

	        Imgproc.equalizeHist(channels.get(0), channels.get(0));

	        Mat result = new Mat();
	        Core.merge(channels,ycrcb);

	        Imgproc.cvtColor(ycrcb,result,Imgproc.COLOR_YCrCb2RGB);

	        return result;
	    } else {
	    	return inputImage;
	    }
	}

}
