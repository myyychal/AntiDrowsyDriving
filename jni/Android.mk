LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

include D:\\AndroidDevelopment\\OpenCV-2.4.9-android-sdk\\sdk\\native\\jni\\OpenCV.mk

LOCAL_MODULE    := anti_drowsy_driving
LOCAL_SRC_FILES := MainFunctions.cpp
LOCAL_SRC_FILES += DetectorFunctions.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)

# Add prebuilt libocr
include $(CLEAR_VARS)

LOCAL_MODULE := libfacialproc_jni
LOCAL_SRC_FILES := libfacialproc_jni.so

include $(PREBUILT_SHARED_LIBRARY)
