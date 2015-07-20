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

#define LOG_TAG "AntiDrowsyDriving::Native"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

extern "C" {

void detectFaceAndDisplay(Mat inputRGBFrame, Mat inputGrayFrame);
void detectEyesAndDisplay(Mat inputRGBFrame, Mat inputGrayFrame);
void equalizeHistogram(Mat inputFrame);
bool findSkinRGB(int R, int G, int B);
bool findSkinYCrCb(float Y, float Cr, float Cb);
bool findSkinHSV(float H, float S, float V);
Mat getSkin(Mat const &src);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier left_eye_cascade;
CascadeClassifier right_eye_cascade;
CascadeClassifier mouth_cascade;
CascadeClassifier nose_cascade;

Mat matOpFlowPrev, matOpFlowThis;
vector<Point> MOPcorners;
vector<Point2f> mMOP2f1, mMOP2f2, mMOP2fptsPrev, mMOP2fptsThis, mMOP2fptsSafe;
vector<uchar> mMOBStatus;
vector<float> mMOFerr;
vector<Point> cornersThis, cornersPrev;
int iGFFTMax = 40;

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_EqualizeHistogram(
		JNIEnv*, jobject, jlong addrGray);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_PrepareFindFace(
		JNIEnv* jEnv, jobject, jstring jFileName);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_PrepareFindEyes(
		JNIEnv* jEnv, jobject, jstring jFileName);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_FindFace(
		JNIEnv*, jobject, jlong addrGray, jlong addrRgba);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_ApplyCLAHE(
		JNIEnv*, jobject, jlong addrSrc, jdouble clipLimit);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_ApplyCLAHEExt(
		JNIEnv*, jobject, jlong addrSrc, jdouble clipLimit, jdouble tileWidth,
		jdouble tileHeight);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_SegmentSkin(
		JNIEnv*, jobject, jlong addrSrc, jlong addrDst);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_FindFeatures(
		JNIEnv*, jobject, jlong addrGray, jlong addrRgb);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_FindCornerHarris(
		JNIEnv*, jobject, jlong addrGray, jlong addrDst);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_CalculateOpticalFlow(
		JNIEnv*, jobject, jlong addrSrc, jlong addrDst);

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_EqualizeHistogram(
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

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_PrepareFindFace(
		JNIEnv* jEnv, jobject, jstring jFileName) {
	const char* jnamestr = jEnv->GetStringUTFChars(jFileName, NULL);
	string stdFileName(jnamestr);
	if (!face_cascade.load(stdFileName)) {
		LOGD("Face cascade file was not loaded");
	};
}

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_PrepareFindEyes(
		JNIEnv* jEnv, jobject, jstring jFileName) {
	const char* jnamestr = jEnv->GetStringUTFChars(jFileName, NULL);
	string stdFileName(jnamestr);
	if (!eyes_cascade.load(stdFileName)) {
		LOGD("Eyes cascade file was not loaded");
	};
}

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_PrepareFindFaceAndOther(
		JNIEnv* jEnv, jobject, jstring jFileName) {
	const char* jnamestr = jEnv->GetStringUTFChars(jFileName, NULL);
	string stdFileName(jnamestr);
	if (!face_cascade.load(stdFileName)) {
		LOGD("Eyes cascade file was not loaded");
	};
	jnamestr = jEnv->GetStringUTFChars(jFileName, NULL);
	string eyesFileName(jnamestr);
	if (!eyes_cascade.load(eyesFileName)) {
		LOGD("Eyes cascade file was not loaded");
	};
}

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_FindFace(
		JNIEnv*, jobject, jlong addrGray, jlong addrRgba) {
	Mat& inputRGBFrame = *(Mat*) addrRgba;
	Mat& inputGrayFrame = *(Mat*) addrGray;
	detectFaceAndDisplay(inputRGBFrame, inputGrayFrame);
}

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_FindEyes(
		JNIEnv*, jobject, jlong addrGray, jlong addrRgba) {
	Mat& inputRGBFrame = *(Mat*) addrRgba;
	Mat& inputGrayFrame = *(Mat*) addrGray;
	detectEyesAndDisplay(inputRGBFrame, inputGrayFrame);
}

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_FindFaceAndOther(
		JNIEnv*, jobject, jlong addrGray, jlong addrRgba) {
	Mat& inputRGBFrame = *(Mat*) addrRgba;
	Mat& inputGrayFrame = *(Mat*) addrGray;
	detectEyesAndDisplay(inputRGBFrame, inputGrayFrame);
}

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_ApplyCLAHE(
		JNIEnv*, jobject, jlong addrSrc, jdouble clipLimit) {
	Mat *src = (Mat*) addrSrc;
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(clipLimit);

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

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_ApplyCLAHEExt(
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

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_SegmentSkin(
		JNIEnv*, jobject, jlong addrSrc, jlong addrDst) {
	Mat& src = *(Mat*) addrSrc;
	Mat& dst = *(Mat*) addrDst;
	dst = getSkin(src);
	erode(dst, dst, Mat(), Point(-1, -1), 2, 1, 1);
}

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_FindFeatures(
		JNIEnv*, jobject, jlong addrGray, jlong addrRgb) {

	Mat& mGr = *(Mat*) addrGray;
	Mat& mRgb = *(Mat*) addrRgb;
	vector<KeyPoint> v;

	FastFeatureDetector detector(8, true);
	detector.detect(mGr, v);
	for (unsigned int i = 0; i < v.size(); i++) {
		const KeyPoint& kp = v[i];
		circle(mRgb, Point(kp.pt.x, kp.pt.y), 5, Scalar(255, 0, 0, 255));
	}

}

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_FindCornerHarris(
		JNIEnv*, jobject, jlong addrGray, jlong addrDst) {
	Mat& mGr = *(Mat*) addrGray;
	Mat& dst_norm_scaled = *(Mat*) addrDst;

	Mat dst, dst_norm;
	dst = Mat::zeros(mGr.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	int thresh = 200;

	/// Detecting corners
	cornerHarris(mGr, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

//	/// Drawing a circle around corners
//	for (int j = 0; j < dst_norm.rows; j++) {
//		for (int i = 0; i < dst_norm.cols; i++) {
//			if ((int) dst_norm.at<float>(j, i) > thresh) {
//				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
//			}
//		}
//	}

	dst.release();
	dst_norm.release();
}

JNIEXPORT void JNICALL Java_com_mpanek_activities_main_MainActivity_CalculateOpticalFlow(
		JNIEnv*, jobject, jlong addrSrc, jlong addrDst) {

	Mat& mRgba = *(Mat*) addrSrc;
	Mat& currentlyUsedFrame = *(Mat*) addrDst;

	if (mMOP2fptsPrev.size() == 0) {

		cvtColor(mRgba, matOpFlowThis, COLOR_RGBA2GRAY);

		matOpFlowThis.copyTo(matOpFlowPrev);

		goodFeaturesToTrack(matOpFlowPrev, MOPcorners, iGFFTMax, 0.05,
				20);

		mMOP2fptsPrev.clear();
		for (int i=0; i<MOPcorners.size(); i++){
			Point2f point = MOPcorners[i];
			mMOP2fptsPrev.push_back(point);
		}

		mMOP2fptsSafe.clear();
		for (int i=0; i<mMOP2fptsPrev.size(); i++){
			mMOP2fptsSafe.push_back((Point2f)mMOP2fptsPrev[i]);
		}

	} else {

		matOpFlowThis.copyTo(matOpFlowPrev);

		cvtColor(mRgba, matOpFlowThis, COLOR_RGBA2GRAY);

		goodFeaturesToTrack(matOpFlowThis, MOPcorners, iGFFTMax, 0.05,
				20);

		mMOP2fptsPrev.clear();
		for (int i=0; i<mMOP2fptsSafe.size(); i++){
			mMOP2fptsPrev.push_back((Point2f)mMOP2fptsSafe[i]);
		}

		mMOP2fptsSafe.clear();
		for (int i=0; i<mMOP2fptsThis.size(); i++){
			mMOP2fptsSafe.push_back((Point2f)mMOP2fptsThis[i]);
		}

	}

	calcOpticalFlowPyrLK(matOpFlowPrev, matOpFlowThis, mMOP2fptsPrev,
			mMOP2fptsThis, mMOBStatus, mMOFerr);

	cornersPrev.clear();
	for (int i=0; i<mMOP2fptsPrev.size(); i++){
		Point point = mMOP2fptsPrev[i];
		cornersPrev.push_back(point);
	}

	cornersThis.clear();
	for (int i=0; i<mMOP2fptsPrev.size(); i++){
		Point point = mMOP2fptsThis[i];
		cornersThis.push_back(point);
	}

	int y = mMOBStatus.size() - 1;

	for (int x = 0; x < y; x++) {
		if (mMOBStatus[x] == 1) {
			Point pt = cornersThis[x];
			Point pt2 = cornersPrev[x];
			circle(currentlyUsedFrame, pt, 5, Scalar(255,0,0), 5);
			line(currentlyUsedFrame, pt, pt2, Scalar(255,0,0), 3);
		}
	}
}

void detectFaceAndDisplay(Mat rgbFrame, Mat grayFrame) {
	vector<Rect> faces;

	float mRelativeFaceSize = 0.2;

	int minFaceSize = 30;
	int heightGray = grayFrame.rows;
	if (heightGray * mRelativeFaceSize > 0) {
		minFaceSize = heightGray * mRelativeFaceSize;
	}

	face_cascade.detectMultiScale(grayFrame, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE,
			Size(minFaceSize, minFaceSize));

	for (int i = 0; i < faces.size(); i++) {
		Rect face = faces[i];
		Point center(face.x + face.width * 0.5, face.y + face.height * 0.5);
		ellipse(rgbFrame, center, Size(face.width * 0.5, face.height * 0.5), 0,
				0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}
}

void detectEyesAndDisplay(Mat rgbFrame, Mat grayFrame) {
	vector<Rect> eyes;

	equalizeHistogram(grayFrame);
	float mRelativeEyesSize = 0.05;

	int minEyesSize = 10;
	int heightGray = grayFrame.rows;
	if (heightGray * mRelativeEyesSize > 0) {
		minEyesSize = heightGray * mRelativeEyesSize;
	}

	eyes_cascade.detectMultiScale(grayFrame, eyes, 1.1, 2, CV_HAAR_SCALE_IMAGE,
			Size(minEyesSize, minEyesSize));

	for (int i = 0; i < eyes.size(); i++) {
		Rect face = eyes[i];
		Point center(face.x + face.width * 0.5, face.y + face.height * 0.5);
		ellipse(rgbFrame, center, Size(face.width * 0.5, face.height * 0.5), 0,
				0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}
}

void equalizeHistogram(Mat inputFrame) {
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

bool findSkinRGB(int R, int G, int B) {
	bool e1 = (R > 95) && (G > 40) && (B > 20)
			&& ((max(R, max(G, B)) - min(R, min(G, B))) > 15)
			&& (abs(R - G) > 15) && (R > G) && (R > B);
	bool e2 = (R > 220) && (G > 210) && (B > 170) && (abs(R - G) <= 15)
			&& (R > B) && (G > B);
	return (e1 || e2);
}

bool findSkinYCrCb(float Y, float Cr, float Cb) {
	bool e3 = Cr <= 1.5862 * Cb + 20;
	bool e4 = Cr >= 0.3448 * Cb + 76.2069;
	bool e5 = Cr >= -4.5652 * Cb + 234.5652;
	bool e6 = Cr <= -1.15 * Cb + 301.75;
	bool e7 = Cr <= -2.2857 * Cb + 432.85;
	return e3 && e4 && e5 && e6 && e7;
}

bool findSkinHSV(float H, float S, float V) {
	return (H < 25) || (H > 230);
}

Mat getSkin(Mat const &src) {
	Mat src_ycrcb, src_hsv;

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

	cvtColor(src, src_ycrcb, CV_RGB2YCrCb);

	src.convertTo(src_hsv, CV_32FC3);
	cvtColor(src_hsv, src_hsv, CV_RGB2HSV);

	normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
			int Y = pix_ycrcb.val[0];
			int Cr = pix_ycrcb.val[1];
			int Cb = pix_ycrcb.val[2];
			// apply ycrcb rule
			bool b = findSkinYCrCb(Y, Cr, Cb);

			Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
			float H = pix_hsv.val[0];
			float S = pix_hsv.val[1];
			float V = pix_hsv.val[2];
			// apply hsv rule
			bool c = findSkinHSV(H, S, V);

			if (b && c) {
				dst.at<unsigned char>(i, j) = 255;
			} else {
				dst.at<unsigned char>(i, j) = 0;
			}

		}
	}
	return dst;
}

}
