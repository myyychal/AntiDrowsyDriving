#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <errno.h>
#include <jni.h>
#include <sys/time.h>
#include <time.h>
#include <android/log.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <queue>

using namespace std;
using namespace cv;

#define LOG_TAG "AntiDrowsyDriving::AlgorithmFunctions"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

extern "C" {

JNIEXPORT void JNICALL Java_com_mpanek_algorithms_general_ClaheAlgorithm_ApplyCLAHEExt(
		JNIEnv*, jobject, jlong addrSrc, jdouble clipLimit, jdouble tileWidth,
		jdouble tileHeight);

JNIEXPORT void JNICALL Java_com_mpanek_algorithms_general_HistogramEqualizationAlgorithm_EqualizeHistogram(
		JNIEnv*, jobject, jlong addrGray);

JNIEXPORT void JNICALL Java_com_mpanek_algorithms_general_ClaheAlgorithm_ApplyCLAHEExt(
		JNIEnv*, jobject, jlong addrSrc, jdouble clipLimit, jdouble tileWidth,
		jdouble tileHeight) {
	Mat *src = (Mat*) addrSrc;

	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(clipLimit);
	Size *tilesSize = new Size(tileWidth, tileHeight);
	clahe->setTilesGridSize(*tilesSize);

	if (src->channels() >= 3) {
		Mat ycrcb;

		cvtColor(*src, ycrcb, CV_RGB2YCrCb);

		vector<Mat> channels;
		split(ycrcb, channels);

		clahe->apply((Mat) channels[0], (Mat) channels[0]);

		merge(channels, ycrcb);

		cvtColor(ycrcb, *src, CV_YCrCb2RGB);

	} else {
		clahe->apply(*src, *src);
	}

}

JNIEXPORT void JNICALL Java_com_mpanek_algorithms_general_HistogramEqualizationAlgorithm_EqualizeHistogram(
		JNIEnv*, jobject, jlong addrGray) {
	Mat& inputFrame = *(Mat*) addrGray;
	if (inputFrame.channels() >= 3) {
		Mat ycrcb;

		cvtColor(inputFrame, ycrcb, CV_RGB2YCrCb);

		vector<Mat> channels;
		split(ycrcb, channels);

		equalizeHist((Mat) channels[0], (Mat) channels[0]);

		merge(channels, ycrcb);

		cvtColor(ycrcb, inputFrame, CV_YCrCb2RGB);

	} else {
		equalizeHist(inputFrame, inputFrame);
	}
}

}


