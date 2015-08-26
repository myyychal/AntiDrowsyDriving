package com.mpanek.utils;

import java.util.ArrayList;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import com.mpanek.constants.DrawingConstants;

public class DrawingUtils {
	
	private static final String TAG = "AntiDrowsyDriving::DrawingUtils";

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
	
	public static void putText(Mat frame, String text, Point startPoint){
		Core.putText(frame, text, startPoint, Core.FONT_HERSHEY_PLAIN, Double.valueOf(2), DrawingConstants.WHITE);
	}
	
	public static void removeAllBorders(Mat frame, int k, int valueToSet) {
		byte buff[] = new byte[(int) (frame.total() * frame.channels())];
		frame.get(0, 0, buff);
		for (int i = 0; i < buff.length; i++) {
			if (i < k * frame.cols() || i > (frame.rows() - k) * frame.cols()) {
				buff[i] = (byte) valueToSet;
			}
		}
		for (int j = 1; j < frame.rows() - k; j++) {
			for (int l = 1; l < frame.cols(); l++) {
				if (l < k || l > frame.cols() - k) {
					buff[(k + j) * frame.cols() + l] = (byte) valueToSet;
				}
			}
		}
		frame.put(0, 0, buff);
	}
	
	public static void removeHorizontalBorders(Mat frame, int k, int valueToSet) {
		byte buff[] = new byte[(int) (frame.total() * frame.channels())];
		frame.get(0, 0, buff);
		for (int i = 0; i < buff.length; i++) {
			if (i < k * frame.cols() || i > (frame.rows() - k) * frame.cols()) {
				buff[i] = (byte) valueToSet;
			}
		}
		frame.put(0, 0, buff);
	}
	
	public static void removeVerticalBorders(Mat frame, int k, int valueToSet) {
		byte buff[] = new byte[(int) (frame.total() * frame.channels())];
		frame.get(0, 0, buff);
		for (int j = 0; j < frame.rows(); j++) {
			for (int l = 0; l < frame.cols(); l++) {
				if (l < k || l > frame.cols() - k) {
					buff[j * frame.cols() + l] = (byte) valueToSet;
				}
			}
		}
		frame.put(0, 0, buff);
	}
}
