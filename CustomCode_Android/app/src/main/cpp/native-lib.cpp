#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "CustomCodeRecognition.h"
using namespace cv;

extern "C" JNIEXPORT jstring JNICALL
Java_customcode_customcode_1android_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_customcode_customcode_1android_MainActivity_CustomCode(JNIEnv *env, jobject instance,
                                                            jlong matAddrInput,
                                                            jlong matAddrResult) {

    // TODO
    Mat &matInput = *(Mat *)matAddrInput;
    Mat &matResult = *(Mat *)matAddrResult;
    cvtColor(matInput, matResult, CV_RGBA2GRAY);

    string result = "";
    CustomCode customcode;
    vector<Point2f> markers;
    customcode.recognition(&matInput, &markers, &result);
    
    return env->NewStringUTF(result.c_str());
}
