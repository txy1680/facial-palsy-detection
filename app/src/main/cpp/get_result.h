#include <jni.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "Backend.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Interpreter.hpp"
#include "Tensor.hpp"

using namespace MNN;
using namespace std;
using namespace cv;

tuple<cv::Mat,std::vector<int>> mnist(Mat image_src, const char* model_name);
