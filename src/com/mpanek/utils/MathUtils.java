package com.mpanek.utils;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import android.R.array;

public class MathUtils {

	private static final String TAG = "AntiDrowsyDriving::MathUtils";

	public static BigDecimal round(float d, int decimalPlace) {
		BigDecimal bd = new BigDecimal(Float.toString(d));
		bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
		return bd;
	}

	public static int findMax(ArrayList<Integer> array) {
		int max = array.get(0);
		for (int i = 1; i < array.size(); i++) {
			if (array.get(i) > max) {
				max = array.get(i);
			}
		}
		return max;
	}

	public static long findMax(ArrayList<Long> array) {
		long max = array.get(0);
		for (int i = 1; i < array.size(); i++) {
			if (array.get(i) > max) {
				max = array.get(i);
			}
		}
		return max;
	}

	public static int findMin(ArrayList<Integer> array) {
		int min = array.get(0);
		for (int i = 1; i < array.size(); i++) {
			if (array.get(i) < min) {
				min = array.get(i);
			}
		}
		return min;
	}

	public static long findMin(ArrayList<Long> array) {
		long min = array.get(0);
		for (int i = 1; i < array.size(); i++) {
			if (array.get(i) < min) {
				min = array.get(i);
			}
		}
		return min;
	}

	public static int normalizeValue(int value, int oldmin, int oldmax, int newmin, int newmax) {
		return newmin + (value - oldmin) * (newmax - newmin) / (oldmax - oldmin);
	}

	public static long normalizeValue(long value, long oldmin, long oldmax, long newmin, long newmax) {
		return newmin + (value - oldmin) * (newmax - newmin) / (oldmax - oldmin);
	}

	public static double normalizeValue(double value, double oldmin, double oldmax, double newmin, double newmax) {
		return newmin + (value - oldmin) * (newmax - newmin) / (oldmax - oldmin);
	}

	public static float normalizeValue(float value, float oldmin, float oldmax, float newmin, float newmax) {
		return newmin + (value - oldmin) * (newmax - newmin) / (oldmax - oldmin);
	}

	public static ArrayList<Integer> applyMedianFilterOnIntegerArray(ArrayList<Integer> arrayList, int medianRange) {
		ArrayList<Integer> filteredList = new ArrayList<Integer>();
		if (medianRange % 2 == 0) {
			medianRange += 1;
		}
		int halfMedianRange = medianRange / 2;
		for (int i = 0; i < halfMedianRange; i++) {
			filteredList.add(arrayList.get(i));
		}
		for (int i = halfMedianRange; i < arrayList.size() - halfMedianRange; i++) {
			ArrayList<Integer> medianCandidatesList = new ArrayList<Integer>(arrayList.subList(i - halfMedianRange, i + halfMedianRange + 1));
			Collections.sort(medianCandidatesList);
			int median = medianCandidatesList.get(halfMedianRange);
			filteredList.add(median);
		}
		for (int i = arrayList.size() - halfMedianRange; i < arrayList.size(); i++) {
			filteredList.add(arrayList.get(i));
		}
		return filteredList;
	}

	public static ArrayList<Long> applyMedianFilterOnLongArray(ArrayList<Long> arrayList, int medianRange) {
		ArrayList<Long> filteredList = new ArrayList<Long>();
		if (medianRange % 2 == 0) {
			medianRange += 1;
		}
		int halfMedianRange = medianRange / 2;
		for (int i = 1; i < halfMedianRange; i++) {
			ArrayList<Long> medianCandidatesList = new ArrayList<Long>(arrayList.subList(i - 1, i + 2));
			Collections.sort(medianCandidatesList);
			long median = medianCandidatesList.get(1);
			filteredList.add(median);
		}
		for (int i = halfMedianRange; i < arrayList.size() - halfMedianRange; i++) {
			ArrayList<Long> medianCandidatesList = new ArrayList<Long>(arrayList.subList(i - halfMedianRange, i + halfMedianRange + 1));
			Collections.sort(medianCandidatesList);
			long median = medianCandidatesList.get(halfMedianRange);
			filteredList.add(median);
		}
		for (int i = arrayList.size() - halfMedianRange; i < arrayList.size()-2; i++) {
			ArrayList<Long> medianCandidatesList = new ArrayList<Long>(arrayList.subList(i - 1, i + 2));
			Collections.sort(medianCandidatesList);
			long median = medianCandidatesList.get(1);
			filteredList.add(median);
		}
		return filteredList;
	}

	public static ArrayList<Long> applyMeanFilterOnLongArray(ArrayList<Long> arrayList, int medianRange) {
		ArrayList<Long> filteredList = new ArrayList<Long>();
		if (medianRange % 2 == 0) {
			medianRange += 1;
		}
		int halfMedianRange = medianRange / 2;
		for (int i = 1; i < halfMedianRange; i++) {
			ArrayList<Long> medianCandidatesList = new ArrayList<Long>(arrayList.subList(i - 1, i + 2));
			filteredList.add(calculateAverage(medianCandidatesList));
		}
		for (int i = halfMedianRange; i < arrayList.size() - halfMedianRange; i++) {
			ArrayList<Long> medianCandidatesList = new ArrayList<Long>(arrayList.subList(i - halfMedianRange, i + halfMedianRange + 1));
			filteredList.add(calculateAverage(medianCandidatesList));
		}
		for (int i = arrayList.size() - halfMedianRange; i < arrayList.size()-2; i++) {
			ArrayList<Long> medianCandidatesList = new ArrayList<Long>(arrayList.subList(i - 1, i + 2));
			filteredList.add(calculateAverage(medianCandidatesList));
		}
		return filteredList;
	}

	public static long calculateAverage(List<Long> array) {
		Long sum = (long) 0;
		if (!array.isEmpty()) {
			for (Long mark : array) {
				sum += mark;
			}
			return (long) (sum.doubleValue() / array.size());
		}
		return sum;
	}
}
