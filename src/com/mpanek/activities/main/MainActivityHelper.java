package com.mpanek.activities.main;

import com.mpanek.constants.ViewModesConstants;

import android.view.MenuItem;

public class MainActivityHelper {

	public static int getViewMode(MenuItem item) {
		int mViewMode = ViewModesConstants.VIEW_MODE_NONE;
		if (item.getGroupId() == 1) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_SNAPDRAGON;
				break;
			case 1:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_JAVA;
				break;
			case 2:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_CPP;
				break;
			case 3:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_CASCADE_JAVA_ASYNC_TASK;
				break;
			case 4:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_JAVA;
				break;
			case 5:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_FACE_COLOR_SEGMENTATION_CPP;
				break;
			}
		} else if (item.getGroupId() == 2) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_JAVA;
				break;
			case 1:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_EYES_CASCADE_CPP;
				break;
			}
		} else if (item.getGroupId() == 5) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_JAVA;
				break;
			case 1:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_GOOD_FEATURES_TO_TRACK_CPP;
				break;
			case 2:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_CORNER_HARRIS_JAVA;
				break;
			case 3:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_CORNER_HARRIS_CPP;
				break;
			case 4:
				mViewMode = ViewModesConstants.VIEW_MODE_OPTICAL_FLOW_JAVA;
				break;
			case 5:
				mViewMode = ViewModesConstants.VIEW_MODE_OPTICAL_FLOW_CPP;
				break;
			case 6:
				mViewMode = ViewModesConstants.VIEW_MODE_STASM;
				break;
			case 7:
				mViewMode = ViewModesConstants.VIEW_MODE_MEAN_VALUES_VERTICAL_PROJECTION_ANALYSIS;
				break;
			case 8:
				mViewMode = ViewModesConstants.VIEW_MODE_INTENSITY_VERTICAL_PROJECTION_ANALYSIS;
			}
		} else if (item.getGroupId() == 6) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_NOSE_CASCADE_JAVA;
				break;
			}
		} else if (item.getGroupId() == 7) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_FIND_ALL;
				break;
			}
		} else if (item.getGroupId() == 8) {
			switch (item.getItemId()) {
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_BIN_STANDARD;
				break;
			case 1:
				mViewMode = ViewModesConstants.VIEW_MODE_BIN_STANDARD_TRUNC;
				break;
			case 2:
				mViewMode = ViewModesConstants.VIEW_MODE_BIN_OTSU;
				break;
			case 3:
				mViewMode = ViewModesConstants.VIEW_MODE_BIN_ADAPTIVE_MEAN;
				break;
			case 4:
				mViewMode = ViewModesConstants.VIEW_MODE_BIN_ADAPTIVE_GAUSSIAN;
				break;
			}
		} else if (item.getGroupId() == 9){
			switch (item.getItemId()){
			case 0:
				mViewMode = ViewModesConstants.VIEW_MODE_CANNY_EDGE;
				break;
			case 1:
				mViewMode = ViewModesConstants.VIEW_MODE_SOBEL_EDGE;
				break;
			case 2:
				mViewMode = ViewModesConstants.VIEW_MODE_LAPLACIAN_EDGE;
				break;
			case 3:
				mViewMode = ViewModesConstants.VIEW_MODE_SOBEL_EDGE_ADVANCED;
				break;
			case 4:
				mViewMode = ViewModesConstants.VIEW_MODE_LAPLACIAN_EDGE_ADVANCED;
				break;
			}
		}
		return mViewMode;
	}

	public static int getHistMode(MenuItem item) {
		int mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_NONE;
		if (item.getGroupId() == 4) {
			switch (item.getItemId()) {
			case 0:
				mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_NONE;
				break;
			case 1:
				mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_HIST_JAVA;
				break;
			case 2:
				mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_HIST_CPP;
				break;
			case 3:
				mEqHistMode = ViewModesConstants.VIEW_MODE_EQ_HIST_CLAHE_CPP;
				break;
			}
		}
		return mEqHistMode;
	}

}
