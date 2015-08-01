package com.mpanek.utils;

import java.util.ArrayList;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

public class DrawingUtils {

	public static void drawRect(Rect rect, Mat imgToDraw, Scalar color, int thickness) {
		Core.rectangle(imgToDraw, rect.tl(), rect.br(), color, thickness);
	}

	public static void drawRect(Rect rect, Mat imgToDraw, Scalar color) {
		if (rect != null) {
			Core.rectangle(imgToDraw, rect.tl(), rect.br(), color, 1);
		}
	}

	public static void drawRects(Rect[] rects, Mat imgToDraw, Scalar color, int thickness) {
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

	public static void drawLines(Point[] points, Mat imgToDraw, Scalar color, int thickness) {
		if (points != null) {
			ArrayList<Point> pointsAsList = new ArrayList<Point>(Arrays.asList(points));
			for (Point pointFromList : pointsAsList) {
				ArrayList<Point> copyList = (ArrayList<Point>) pointsAsList.clone();
				copyList.remove(pointFromList);
				for (Point anotherPointFromList : copyList) {
					Core.line(imgToDraw, pointFromList, anotherPointFromList, color, thickness);
				}
			}
		}
	}

	public static void drawLines(Point[] points, Mat imgToDraw, Scalar color) {
		drawLines(points, imgToDraw, color, 1);
	}

	public static void drawLinesFromRectanglesCentres(Rect[] rects, Mat imgToDraw, Scalar color, int thickness) {
		if (rects != null) {
			ArrayList<Point> centrePoints = new ArrayList<Point>();
			for (Rect rect : rects){
				centrePoints.add(VisualUtils.getCentrePoint(rect));
			}
			Point[] centrePointsArray = centrePoints.toArray(new Point[centrePoints.size()]);
			drawLines(centrePointsArray, imgToDraw, color, thickness);
		}
	}
	
	public static void drawLinesFromRectanglesCentres(Rect[] rects, Mat imgToDraw, Scalar color){
		drawLinesFromRectanglesCentres(rects, imgToDraw, color, 1);
	}

}
