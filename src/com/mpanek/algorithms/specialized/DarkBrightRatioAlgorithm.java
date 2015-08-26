package com.mpanek.algorithms.specialized;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.util.Log;

import com.mpanek.algorithms.general.BinarizationAlgorithm;
import com.mpanek.algorithms.general.ClaheAlgorithm;
import com.mpanek.algorithms.general.HistogramEqualizationAlgorithm;
import com.mpanek.constants.DrawingConstants;
import com.mpanek.utils.MathUtils;

public class DarkBrightRatioAlgorithm {
	
	private static final String TAG = "AntiDrowsyDriving::DarkBrightRatioAlgorithm";

	ClaheAlgorithm claheAlgorithm;
	HistogramEqualizationAlgorithm histogramEqualizationAlgorithm;
	BinarizationAlgorithm binarizationAlgorithm;

	long whitePixels;
	long blackPixesls;
	
	float meanValuePixels;
	
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

	public void claheEqualizeSimpleBinarizeAndOpenOperation(Mat frame) {
		claheAlgorithm.process(frame);
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		binarizationAlgorithm.standardBinarization(frame);

		performOpenOperation(frame);
	}
	
	public void claheEqualizeAdaptiveBinarizeAndOpenOperation(Mat frame) {
		claheAlgorithm.process(frame);
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		binarizationAlgorithm.adaptiveMeanBinarization(frame);
		performCloseOperation(frame);
	}
	
	public void countMeanBlackAndWhitePixels(Mat frame){
		Scalar meanValues = Core.mean(frame);
		this.meanValuePixels = (float) meanValues.val[0];
		this.whitePixels = Core.countNonZero(frame);
		this.blackPixesls = frame.width() * frame.height() - this.whitePixels;
	}
	
	public ArrayList<Long> countIntensityVerticalProjection(Mat frame){
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		byte buff[] = new byte[(int) (frame.total() * frame.channels())];
		frame.get(0, 0, buff);
		ArrayList<Long> intensityInRows = new ArrayList<Long>();
		long rowSum = 0;
		for (int i=0; i<buff.length; i++){
			if ((i+1) % frame.width() == 0){
				intensityInRows.add(rowSum);
				rowSum = 0;
			}
			rowSum += buff[i] & 0xff;
		}
		return intensityInRows;
	}
	
	public ArrayList<Long> countIntensityHorizontalProjection(Mat frame){
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		byte buff[] = new byte[(int) (frame.total() * frame.channels())];
		frame.get(0, 0, buff);
		ArrayList<Long> intensityInCols = new ArrayList<Long>(Collections.nCopies(frame.width(), 0l));
		int j=0;
		for (int i=0; i<buff.length; i++){
			if ((i+1) % frame.width() == 0){
				j = 0;
			}
			long valueToSet = intensityInCols.get(j) + (buff[i] & 0xff);
			intensityInCols.set(j, valueToSet);
			j++;
		}
		return intensityInCols;
	}
	
	public ArrayList<Long> countMeanVerticalProjection(Mat frame){
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		byte buff[] = new byte[(int) (frame.total() * frame.channels())];
		frame.get(0, 0, buff);
		ArrayList<Long> meanInRows = new ArrayList<Long>();
		long rowSum = 0;
		for (int i=0; i<buff.length; i++){
			if ((i+1) % frame.width() == 0){
				meanInRows.add((long) (rowSum/frame.width()));
				rowSum = 0;
			}
			rowSum += buff[i] & 0xff;
		}
		return meanInRows;
	}
	
	public ArrayList<Long> countMeanHorizontalProjection(Mat frame){
		if (frame.channels() == 3) {
			Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2GRAY);
		}
		byte buff[] = new byte[(int) (frame.total() * frame.channels())];
		frame.get(0, 0, buff);
		ArrayList<Long> meanInRows = new ArrayList<Long>(Collections.nCopies(frame.width(), 0l));
		int j=0;
		for (int i=0; i<buff.length; i++){
			if ((i+1) % frame.width() == 0){
				j = 0;
			}
			long valueToSet = meanInRows.get(j) + (buff[i] & 0xff);
			meanInRows.set(j, valueToSet);
			j++;
		}
		for (int i=0; i<meanInRows.size(); i++){
			meanInRows.set(i, (Long)(meanInRows.get(i)/frame.height()));
		}
		return meanInRows;
	}
	
	public ArrayList<Long> normalizeAndDrawVerticalProjectionAnalysisArrays(Mat frame, ArrayList<Long> array){
		ArrayList<Long> vpaValuesArrayList = array;
		ArrayList<Long> normalizedPlotValues = new ArrayList<Long>();
		long oldMaxValue = MathUtils.findMax(vpaValuesArrayList);
		long oldMinValue = MathUtils.findMin(vpaValuesArrayList);
		long newMin = 0;
		long newMax = frame.width();
		for (Long value : vpaValuesArrayList) {
			normalizedPlotValues.add(MathUtils.normalizeValue(value, oldMinValue, oldMaxValue, newMin, newMax));
		}
		for (int i = 0; i < normalizedPlotValues.size(); i++) {
			Core.circle(frame, new Point(normalizedPlotValues.get(i), i), 1, DrawingConstants.PINK);
		}
		return normalizedPlotValues;
	}
	
	public ArrayList<Long> normalizeAndDrawVerticalProjectionAnalysisArrays(Mat frame, ArrayList<Long> array, long normalizeMin, long normalizeMax){
		ArrayList<Long> vpaValuesArrayList = array;
		ArrayList<Long> normalizedPlotValues = new ArrayList<Long>();
		long newMin = 0;
		long newMax = frame.width();
		for (Long value : vpaValuesArrayList) {
			normalizedPlotValues.add(MathUtils.normalizeValue(value, normalizeMin, normalizeMax, newMin, newMax));
		}
		for (int i = 0; i < normalizedPlotValues.size(); i++) {
			Core.circle(frame, new Point(normalizedPlotValues.get(i), i), 1, DrawingConstants.PINK);
		}
		return normalizedPlotValues;
	}
	
	public ArrayList<Long> normalizeAndDrawHorizontalProjectionAnalysisArrays(Mat frame, ArrayList<Long> array){
		ArrayList<Long> vpaValuesArrayList = array;
		ArrayList<Long> normalizedPlotValues = new ArrayList<Long>();
		long oldMaxValue = MathUtils.findMax(vpaValuesArrayList);
		long oldMinValue = MathUtils.findMin(vpaValuesArrayList);
		long newMin = 0;
		long newMax = frame.height();
		for (Long value : vpaValuesArrayList) {
			normalizedPlotValues.add(MathUtils.normalizeValue(value, oldMinValue, oldMaxValue, newMin, newMax));
		}
		for (int i = 0; i < normalizedPlotValues.size(); i++) {
			Core.circle(frame, new Point(i, normalizedPlotValues.get(i)), 1, DrawingConstants.PINK);
		}
		return normalizedPlotValues;
	}
	
	public ArrayList<Long> normalizeAndDrawHorizontalProjectionAnalysisArrays(Mat frame, ArrayList<Long> array, long normalizeMin, long normalizeMax){
		ArrayList<Long> vpaValuesArrayList = array;
		ArrayList<Long> normalizedPlotValues = new ArrayList<Long>();
		long newMin = 0;
		long newMax = frame.height();
		for (Long value : vpaValuesArrayList) {
			normalizedPlotValues.add(MathUtils.normalizeValue(value, normalizeMin, normalizeMax, newMin, newMax));
		}
		for (int i = 0; i < normalizedPlotValues.size(); i++) {
			Core.circle(frame, new Point(i, normalizedPlotValues.get(i)), 1, DrawingConstants.PINK);
		}
		return normalizedPlotValues;
	}
	
	public void performOpenOperation(Mat frame){
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * erosionSize + 1, 2 * erosionSize + 1));
		Imgproc.erode(frame, frame, element);
		Imgproc.dilate(frame, frame, element);
	}
	
	public void performCloseOperation(Mat frame){
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * erosionSize + 1, 2 * erosionSize + 1));
		Imgproc.dilate(frame, frame, element);
		Imgproc.erode(frame, frame, element);
	}
	
	public void performErodeOperation(Mat frame){
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, new Size(2 * erosionSize + 1, 2 * erosionSize + 1));
		Imgproc.erode(frame, frame, element);
	}

	public void performDilateOperation(Mat frame){
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE, new Size(2 * erosionSize + 1, 2 * erosionSize + 1));
		Imgproc.dilate(frame, frame, element);
	}
	
	public void performOpenOperation(Mat frame, int elemSize){
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * elemSize + 1, 2 * elemSize + 1));
		Imgproc.erode(frame, frame, element);
		Imgproc.dilate(frame, frame, element);
	}
	
	public void performCloseOperation(Mat frame, int elemSize){
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * elemSize + 1, 2 * elemSize + 1));
		Imgproc.dilate(frame, frame, element);
		Imgproc.erode(frame, frame, element);
	}
	
	public void performErodeOperation(Mat frame, int elemSize){
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, new Size(2 * elemSize + 1, 2 * elemSize + 1));
		Imgproc.erode(frame, frame, element);
	}

	public void performDilateOperation(Mat frame, int elemSize){
		Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE, new Size(2 * elemSize + 1, 2 * elemSize + 1));
		Imgproc.dilate(frame, frame, element);
	}
	
	public void fillWhiteSpots(Mat frame, int rangeFrom, int rangeTo, int horizontalBorders){
		byte buff[] = new byte[(int) (frame.total() * frame.channels())];
		frame.get(0, 0, buff);
		int j=0;
		for (int i=0; i<buff.length; i++){
			if ((i+1) % frame.width() == 0){
				j++;
			}
			if ((buff[i] & 0xff) >= rangeFrom && (buff[i] & 0xff) <= rangeTo
					&& j > horizontalBorders && j< frame.height() - horizontalBorders){
				buff[i] = (byte) ((buff[i] & 0xff) - 50);
			}
		}
		frame.put(0,0,buff);
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

	public long getWhitePixels() {
		return whitePixels;
	}

	public long getBlackPixesls() {
		return blackPixesls;
	}

	public float getMeanValuePixels() {
		return meanValuePixels;
	}

	public long getAllPixelsCount(){
		return this.whitePixels + this.blackPixesls;
	}

}
