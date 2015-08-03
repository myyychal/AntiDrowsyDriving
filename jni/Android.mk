LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_INSTALL_MODULES:=on

include D:\\AndroidDevelopment\\OpenCV-2.4.9-android-sdk\\sdk\\native\\jni\\OpenCV.mk

OPENCV_CAMERA_MODULES:=on

LOCAL_MODULE    := anti_drowsy_driving
#LOCAL_SRC_FILES := MainFunctions.cpp
#LOCAL_SRC_FILES += DetectorFunctions.cpp
#LOCAL_SRC_FILES += demo.cpp
LOCAL_FILE_LIST = $(wildcard $(LOCAL_PATH)/stasm/*.cpp) \
				  $(wildcard $(LOCAL_PATH)/stasm/MOD_1/*.cpp) \
				  $(wildcard $(LOCAL_PATH)/*.cpp)

LOCAL_SRC_FILES := $(LOCAL_FILE_LIST:$(LOCAL_PATH)/%=%)
LOCAL_LDLIBS +=  -llog -ldl
#LOCAL_CFLAGS += -O3
#LOCAL_LDFLAGS += -O3

include $(BUILD_SHARED_LIBRARY)

# Add prebuilt libocr
include $(CLEAR_VARS)

LOCAL_MODULE := libfacialproc_jni
LOCAL_SRC_FILES := libfacialproc_jni.so

include $(PREBUILT_SHARED_LIBRARY)