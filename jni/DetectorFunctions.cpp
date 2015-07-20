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

#define LOG_TAG "AntiDrowsyDriving::MainFunctions"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

extern "C" {
}
