#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>
#include "get_result.h"
#include "stdio.h"
#include "stdlib.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_opencvdemo_MainActivity_mnistJNI (JNIEnv *env, jobject obj, jobject bitmap, jstring jstr){

    AndroidBitmapInfo info;
    void *pixels;
    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        Mat temp(info.height, info.width, CV_8UC4, pixels);
        Mat temp2 = temp.clone();
        //将jstring类型转换成C++里的const char*类型
        const char *path = env->GetStringUTFChars(jstr, 0);

        Mat RGB;
        //先将图像格式由BGRA转换成RGB，不然识别结果不对
        cvtColor(temp2, RGB, COLOR_RGBA2RGB);
        std::vector<std::string> s;
        Mat result;
        std::vector<int> switch_result;
        tuple<cv::Mat,std::vector<int>> tup;
        //调用之前定义好的mnist()方法，识别文字图像
        tup = mnist(RGB, path);
        switch_result=std::get<1>(tup);
        for(int i=0;i<switch_result.size();i++)
        {
            switch (switch_result[i])
            {
                case 0:
                    s.insert(s.end(),"眼部轻微");
                    break;
                case 1:
                    s.insert(s.end(),"嘴部轻微");
                    break;
                case 2:
                    s.insert(s.end(),"眼部严重");
                    break;
                case 3:
                    s.insert(s.end(),"嘴部严重");
                    break;
                case 4:
                    s.insert(s.end(),"眼部正常");
                    break;
                case 5:
                    s.insert(s.end(),"嘴部正常");
                    break;
            }
        }

        //将图像转回RGBA格式，Android端才可以显示
        Mat show(info.height, info.width, CV_8UC4, pixels);
        cvtColor(std::get<0>(tup), temp, COLOR_RGB2RGBA);
        //将int类型的识别结果转成jstring类型，并返回
//        string re_reco = to_string(s);
        string s1="诊断结果：";
        for(int k=0;k<s.size();k++)
        {
            s1=s1+" "+s[k];
        }

        string re_reco=s1;
        const char* ss = re_reco.c_str();
        char cap[50];
        strcpy(cap, ss);
        return (env)->NewStringUTF(cap);;

    } else {
        Mat temp(info.height, info.width, CV_8UC2, pixels);

    }
    AndroidBitmap_unlockPixels(env, bitmap);
}
