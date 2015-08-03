#include <jni.h>
#include <android/log.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "stasm/stasm_lib.h"
#include "stasm/stasm_landmarks.h"

using namespace std;
using namespace cv;

#define LOG_TAG "AntiDrowsyDriving::Stasm"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

extern "C" {

JNIEXPORT jintArray JNICALL Java_com_mpanek_activities_main_MainActivity_FindFaceLandmarks(
		JNIEnv* env, jobject obj, jlong addrGray) {

	static const char* path =
			"/data/data/com.mpanek.activities.main/app_stasm/testface.jpg";
	Mat* mGray = (Mat*) addrGray;

	jintArray arrayOfLandmarks = env->NewIntArray(2 * stasm_NLANDMARKS);
	jint* output = env->GetIntArrayElements(arrayOfLandmarks, NULL);

	double start = clock();

	if (!mGray->data) {
		LOGD("Cannot load path");

		env->ReleaseIntArrayElements(arrayOfLandmarks, output, 0);
		return arrayOfLandmarks;
	}

	int foundface;
	float landmarks[2 * stasm_NLANDMARKS] = { 0 };

	if (!stasm_search_single(&foundface, landmarks, (const char*) mGray->data,
			mGray->cols, mGray->rows, path,
			"/data/data/com.mpanek.activities.main/app_stasm/")) {
		LOGD("Error in stasm_search_single %s", stasm_lasterr());


		env->ReleaseIntArrayElements(arrayOfLandmarks, output, 0);
		return arrayOfLandmarks;
	}

	if (!foundface) {
		LOGD("Face not found");

		env->ReleaseIntArrayElements(arrayOfLandmarks, output, 0);
		return arrayOfLandmarks;
	} else {
		LOGD("Face was found");

		stasm_force_points_into_image(landmarks, mGray->cols, mGray->rows);
		double asmTime = double(clock() - start) / CLOCKS_PER_SEC;
		LOGD("Time:%.3fsec", asmTime);

		for (int i = 0; i < stasm_NLANDMARKS; i++) {
			output[2 * i] = cvRound(landmarks[2 * i]);
			output[2 * i + 1] = cvRound(landmarks[2 * i + 1]);
		}

		env->ReleaseIntArrayElements(arrayOfLandmarks, output, 0);
		return arrayOfLandmarks;
	}
}

JNIEXPORT jintArray JNICALL Java_com_mpanek_activities_main_MainActivity_AddFaceLandmarks(
		JNIEnv* env, jobject obj, jlong addrGray, jfloatArray pinnedPoints) {

	static const char* path =
			"/data/data/com.mpanek.activities.main/app_stasm/testface.jpg";
	Mat* mGray = (Mat*) addrGray;

	LOGD("mGray width: %d", mGray->cols);
	LOGD("mGray rows: %d", mGray->rows);

	jintArray arrayOfLandmarks = env->NewIntArray(2 * stasm_NLANDMARKS);
	jint* output = env->GetIntArrayElements(arrayOfLandmarks, NULL);

	double start = clock();

	if (!mGray->data) {
		LOGD("Cannot load path");

		mGray->release();
		env->ReleaseIntArrayElements(arrayOfLandmarks, output, 0);
		return arrayOfLandmarks;
	}

	float landmarks[2 * stasm_NLANDMARKS] = { 0 };

	float* pinnedPointsC = (float *)env->GetFloatArrayElements(pinnedPoints, 0);

    float pinned[2 * stasm_NLANDMARKS] = {0};

    pinned[L_LEyeOuter*2]      = pinnedPointsC[0] + 2;
    pinned[L_LEyeOuter*2+1]    = pinnedPointsC[1];
    pinned[L_REyeOuter*2]      = pinnedPointsC[2] - 2;
    pinned[L_REyeOuter*2+1]    = pinnedPointsC[3];
    pinned[L_CNoseTip*2]       = pinnedPointsC[4];
    pinned[L_CNoseTip*2+1]     = pinnedPointsC[5];
    pinned[L_LMouthCorner*2]   = pinnedPointsC[6];
    pinned[L_LMouthCorner*2+1] = pinnedPointsC[7];
    pinned[L_RMouthCorner*2]   = pinnedPointsC[8];
    pinned[L_RMouthCorner*2+1] = pinnedPointsC[9];

	stasm_init("/data/data/com.mpanek.activities.main/app_stasm/", 0);

	if (!stasm_search_pinned(landmarks, pinned,(const char*) mGray->data,
			mGray->cols, mGray->rows, path)) {
		LOGD("Error in stasm_search_pinned %s", stasm_lasterr());

		env->ReleaseIntArrayElements(arrayOfLandmarks, output, 0);
		return arrayOfLandmarks;
	}

	stasm_force_points_into_image(landmarks, mGray->cols, mGray->rows);
	double asmTime = double(clock() - start) / CLOCKS_PER_SEC;
	LOGD("Time:%.3fsec", asmTime);

	for (int i = 0; i < stasm_NLANDMARKS; i++) {
		output[2 * i] = cvRound(landmarks[2 * i]);
		output[2 * i + 1] = cvRound(landmarks[2 * i + 1]);
	}

	LOGD("mGray width: %d", mGray->cols);
	LOGD("mGray rows: %d", mGray->rows);

	env->ReleaseIntArrayElements(arrayOfLandmarks, output, 0);

	return arrayOfLandmarks;
}
}

