package com.mpanek.utils;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

public class DrawingUtils {

	public static void drawRect(Rect rect, Mat imgToDraw, Scalar color,
			int thickness) {
		Core.rectangle(imgToDraw, rect.tl(), rect.br(), color, thickness);
	}

	public static void drawRect(Rect rect, Mat imgToDraw, Scalar color) {
		if (rect != null) {
			Core.rectangle(imgToDraw, rect.tl(), rect.br(), color, 3);
		}
	}

	public static void drawRects(Rect[] rects, Mat imgToDraw, Scalar color,
			int thickness) {
		for (Rect rect : rects) {
			drawRect(rect, imgToDraw, color, thickness);
		}
	}

	public static void drawRects(Rect[] rects, Mat imgToDraw, Scalar color) {
		if (rects != null) {
			for (Rect rect : rects) {
				drawRect(rect, imgToDraw, color);
			}
		}
	}

}
