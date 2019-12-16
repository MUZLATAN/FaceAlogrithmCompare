//Created by Jack Yu
#include "opencv2/imgproc/imgproc.hpp"
#include "gtest/gtest.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "MTCNN.h"
#include <chrono>
using namespace cv;
using namespace std;
std::time_t get_timestamp_m()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    std::time_t timestamp = tmp.count();
    return timestamp;
}
int main(int argc, char * argv[])
{
    MTCNN detector("/home/z/workspace/Fast_MTCNN/model");

//    //读取待识别图片路径
//    cv::String pattern="/home/z/data/secret/37personimages";
//    std::vector<cv::String> result;
//    cv::glob(pattern, result,true);
//    CV_Assert(result.size() > 0);

    VideoCapture cap("/home/z/Desktop/unnormal.webm");

    Mat img;

    float factor = 0.709f;
    float threshold[3] = { 0.7f, 0.6f, 0.6f };
    int minSize = 12;

//    for (size_t j = 0; j < result.size(); ++j)
    time_t start = get_timestamp_m();
    while (cap.read(img))
    {
//        img = imread(result[j]);
        vector<FaceInfo> faceInfo = detector.Detect_mtcnn(img, minSize, threshold, factor, 3);
//        for (size_t i = 0; i < faceInfo.size(); i++) {
//            int x = (int) faceInfo[i].bbox.xmin;
//            int y = (int) faceInfo[i].bbox.ymin;
//            int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
//            int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
//            cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
//        }
//        cv::imwrite("test.png", img);
//        cv::imshow("image", img);
//        cv::waitKey(10);
    }
    time_t end = get_timestamp_m();
    cout<<"time cost is: "<<end-start<<endl;
    return 0;
}