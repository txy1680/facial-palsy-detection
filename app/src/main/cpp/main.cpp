//
// Created by tan on 2021/7/23.
//
#include <iostream>
#include <string>
#include <vector>
#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <opencv2/opencv.hpp>

#include "Yolo.h"

void show_shape(std::vector<int> shape)
{
    std::cout<<shape[0]<<" "<<shape[1]<<" "<<shape[2]<<" "<<shape[3]<<" "<<shape[4]<<" "<<std::endl;

}

void scale_coords(std::vector<BoxInfo> &boxes, int w_from, int h_from, int w_to, int h_to)
{
    float w_ratio = float(w_to)/float(w_from);
    float h_ratio = float(h_to)/float(h_from);


    for(auto &box: boxes)
    {
        box.x1 *= w_ratio;
        box.x2 *= w_ratio;
        box.y1 *= h_ratio;
        box.y2 *= h_ratio;
    }
    return ;
}

tuple<cv::Mat,std::vector<int>>draw_box(cv::Mat & cv_mat, std::vector<BoxInfo> &boxes, const std::vector<std::string> &labels)
{
    int CNUM = 80;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
        rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    std::vector<int> return_lable;

    for(auto box : boxes)
    {
        int width = box.x2-box.x1;
        int height = box.y2-box.y1;
        int area=width*height;
        if (area>1000)
        {
            int id = box.id;
            cv::Point p = cv::Point(box.x1, box.y1);
            cv::Rect rect = cv::Rect(box.x1, box.y1, width, height);
            cv::rectangle(cv_mat, rect, randColor[box.label]);
            string text = labels[box.label] + ":" + std::to_string(box.score) + " ID:" + std::to_string(id);
            cv::putText(cv_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1, randColor[box.label]);
            return_lable.insert(return_lable.end(),box.label);
        }

    }
    auto tup=std::make_tuple(cv_mat,return_lable);
    return tup;
}

tuple<cv::Mat,std::vector<int>> mnist(cv::Mat image_src, const char* model_name)
{
    const char *pchPath=model_name;
    int num_classes=6;
    std::vector<YoloLayerData> yolov5ss_layers{
            {"397",    32, {{116, 90}, {156, 198}, {373, 326}}},
            {"458",    16, {{30,  61}, {62,  45},  {59,  119}}},
            {"519", 8,  {{10,  13}, {16,  30},  {33,  23}}},
    };
    std::vector<YoloLayerData> & layers = yolov5ss_layers;
    std::vector<std::string> labels{"SlightPalsy_Eyes","SlightPalsy_Mouth","StrongPalsy_Eyes",
                                    "StrongPalsy_Mouth","Normal_Eyes","Normal_Mouth"};

    int net_size =640;


    // auto revertor = std::unique_ptr<Revert>(new Revert(model_name.c_str()));
    // revertor->initialize();
    // auto modelBuffer      = revertor->getBuffer();*
    // const auto bufferSize = revertor->getBufferSize();
    // auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    // revertor.reset();

    std::shared_ptr<MNN::Interpreter> net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(pchPath));


    MNN::ScheduleConfig config;
    config.type=MNN_FORWARD_AUTO;
    MNN::Session *session = net->createSession(config);;
    BoxInfo box;
    int INPUT_SIZE = 640;
    int i;
    cv::Mat src_img=image_src;
    cv::Mat image;
    cv::Mat image1;
    cv::resize(src_img, image, cv::Size(INPUT_SIZE, INPUT_SIZE));
    image1=image;
    // preprocessing
    image.convertTo(image, CV_32FC3);
    // image = (image * 2 / 255.0f) - 1;
    image = image /255.0f;

    // wrapping input tensor, convert nhwc to nchw
    std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    std::memcpy(nhwc_data, image.data, nhwc_size);

    auto inputTensor = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // run network
    net->runSession(session);

    // get output data
    std::string output_tensor_name0 = layers[2].name ;
    std::string output_tensor_name1 = layers[1].name ;
    std::string output_tensor_name2 = layers[0].name ;



    MNN::Tensor *tensor_scores  = net->getSessionOutput(session, output_tensor_name0.c_str());
    MNN::Tensor *tensor_boxes   = net->getSessionOutput(session, output_tensor_name1.c_str());
    MNN::Tensor *tensor_anchors = net->getSessionOutput(session, output_tensor_name2.c_str());

    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    MNN::Tensor tensor_anchors_host(tensor_anchors, tensor_anchors->getDimensionType());

    tensor_scores->copyToHostTensor(&tensor_scores_host);
    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    tensor_anchors->copyToHostTensor(&tensor_anchors_host);

    std::vector<BoxInfo> result;
    std::vector<BoxInfo> boxes;

    yolocv::YoloSize yolosize = yolocv::YoloSize{INPUT_SIZE,INPUT_SIZE};

    float threshold = 0.8;
    float nms_threshold = 0.45;

    show_shape(tensor_scores_host.shape());
    show_shape(tensor_boxes_host.shape());
    show_shape(tensor_anchors_host.shape());


    boxes = decode_infer(tensor_scores_host, layers[2].stride,  yolosize, net_size, num_classes, layers[2].anchors, threshold);
    result.insert(result.begin(), boxes.begin(), boxes.end());

    boxes = decode_infer(tensor_boxes_host, layers[1].stride,  yolosize, net_size, num_classes, layers[1].anchors, threshold);
    result.insert(result.begin(), boxes.begin(), boxes.end());

    boxes = decode_infer(tensor_anchors_host, layers[0].stride,  yolosize, net_size, num_classes, layers[0].anchors, threshold);
    result.insert(result.begin(), boxes.begin(), boxes.end());

    nms(result, nms_threshold);
    box=result[0];
    std::tuple<cv::Mat,std::vector<int>> t1;
    t1 = draw_box(image_src, result, labels);
    return t1;

}




// std::shared_ptr<MNN::Interpreter> net =
//     std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(fileName));
// if (nullptr == net) {
//     return 0;
// }
// // Must call it before createSession.
// // If ".tempcache" file does not exist, Interpreter will go through the
// // regular initialization procedure, after which the compiled model files
// // will be written  to ".tempcache".
// // If ".tempcache" file exists, the Interpreter will be created from the
// // cache.
// net->setCacheFile(".tempcache");

// MNN::ScheduleConfig config;
// // Creates the session after you've called setCacheFile.
// MNN::Session* session = net->createSession(config);
