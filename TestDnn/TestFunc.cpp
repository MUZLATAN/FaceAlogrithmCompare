//
// Created by z on 2019/11/15.
//


#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/face/facemark.hpp"
#include "opencv2/face.hpp"

using namespace std;
using namespace cv;
using namespace dnn;
std::time_t get_timestamp()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    std::time_t timestamp = tmp.count();
    return timestamp;
}

void test1()
{
    const size_t inWidth = 300;
    const size_t inHeight = 300;
    const double inScaleFactor = 1.0;
    const cv::Scalar meanVal(104.0, 117.0, 123.0);

    float min_confidence = 0.5;
    std::string modelConfiguration = "/home/z/data/OpenCVData/deploy_lowres.prototxt";
    std::string modelBinary = "/home/z/data/OpenCVData/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    //初始化网络
    Net net = readNetFromCaffe(modelConfiguration, modelBinary);

    //
    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "prototxt:   " << modelConfiguration << endl;
        cerr << "caffemodel: " << modelBinary << endl;
        cerr << "Models are available here:" << endl;
        cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;
        cerr << "or here:" << endl;
        cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;
        exit(-1);
    }

    // 尝试opencl 加速
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    cv::String pattern="/media/z/fa0b9259-72a5-444c-8b3a-0bfd6b935fcf/home/z/data/myface";
    std::vector<cv::String> result;
    cv::glob(pattern, result,true);
    time_t start = get_timestamp();
    int right_number = 0;
    for (size_t i = 0; i < result.size(); ++i)
    {
        cv::Mat img = cv::imread(result[i]);
        if (img.empty())
        {
            cerr<<"cann't open the image";
            break;
        }

        if (img.channels() == 4)
            cvtColor(img, img, cv::COLOR_BGRA2BGR);
        {

            cv::Mat inputBlob = cv::dnn::blobFromImage(img, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false,
                                                       false);

            //
            net.setInput(inputBlob, "data");

            cv::Mat detection = net.forward("detection_out");
            cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());//

        std::vector<double> layersTimings;

        ostringstream ss;

        putText(img, ss.str(), cv::Point(20, 20), 0, 0.5, cv::Scalar(0, 0, 255));

        float confidenceThreshold = min_confidence;
        for (int j = 0; j < detectionMat.rows; j++)
        {
            float confidence = detectionMat.at<float>(j, 2);      //第二列存放可信度

            if (confidence > confidenceThreshold)//满足阈值条件
            {
                right_number++;
                //存放人脸所在的图像中的位置信息
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(j, 3) * img.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(j, 4) * img.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(j, 5) * img.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(j, 6) * img.rows);

                cv::Rect object((int)xLeftBottom, (int)yLeftBottom,//定义一个矩形区域（x,y,w,h)
                                (int)(xRightTop - xLeftBottom),
                                (int)(yRightTop - yLeftBottom));
//                Mat tmpimg = img(object);
//                cv::imwrite("/home/z/Desktop/myface/"+to_string(i)+".jpg", tmpimg);
//                cv::rectangle(img, object, cv::Scalar(0, 255, 0));//画个边框

//                ss.str("");
//                ss << confidence;
//                cv::String conf(ss.str());
//                cv::String label = "Face: " + conf;
//                int baseLine = 0;
//                cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//                cv::rectangle(img, cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255),0,8,0);
//                putText(img, label, cv::Point(xLeftBottom, yLeftBottom),
//                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }
//            cv::imshow("detections", img);cv::waitKey(10);
        }
    }

    time_t end = get_timestamp();
    cout<<"time cost is:"<<end-start<<endl;
    cout<<"the number is: "<<right_number<<endl;
}

std::vector<Rect> detectFaces(cv::dnn::Net net, cv::Mat img)
{


    cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1, cv::Size(192, 144), cv::Scalar(104.0, 117.0, 123.0), false,false);

    //　值得研究
    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());//n*7矩阵 表示人脸的个数

    //vector<Rect>  用来存放人脸矩形位置
    std::vector<Rect> faces;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);      //第二列存放可信度

        if (confidence > 0.5)//满足阈值条件
        {
            //存放人脸所在的图像中的位置信息
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);


            cv::Rect object((int)xLeftBottom, (int)yLeftBottom,//定义一个矩形区域（x,y,w,h)
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

//            cv::rectangle(img, object, cv::Scalar(0, 255, 0));//画个边框
//            cv::imshow("",img);
//            cv::waitKey(100);
            faces.push_back(object);
        }
    }
    return faces;

}

cv::Mat  face2vec(Net net, cv::Mat frame, const double inScaleFactor, const size_t inWidth, const size_t inHeight, const cv::Scalar meanVal)
{
    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true,false);

    CV_Assert(!inputBlob.empty());
    net.setInput(inputBlob);
    cv::Mat detection = net.forward().clone();
    CV_Assert(!detection.empty());

    return detection;
}
std::string getShortName(std::string Name)
{
    size_t idx1 = Name.find_last_of('/');
    size_t idx2 = Name.find_last_of('.');
    return Name.substr(idx1+1,idx2-idx1-1);
}
cv::Mat face_alignment(Mat face, Point left, Point right, Rect roi) {
    int offsetx = roi.x;
    int offsety = roi.y;

    // 计算中心位置
    int cx = roi.width / 2;
    int cy = roi.height / 2;

    // 计算角度
    int dx = right.x - left.x;
    int dy = right.y - left.y;
    double degree = 180 * ((atan2(dy, dx)) / CV_PI);

    // 旋转矩阵计算
    Mat M = getRotationMatrix2D(Point2f(cx, cy), degree, 1.0);
    Point2f center(cx, cy);
    Rect bbox = RotatedRect(center, face.size(), degree).boundingRect();
    M.at<double>(0, 2) += (bbox.width / 2.0 - center.x);
    M.at<double>(1, 2) += (bbox.height / 2.0 - center.y);

    // 对齐
    Mat result;
    warpAffine(face, result, M, bbox.size());
//    imshow("face-alignment", result);
    return result;
}
std::map<std::string,cv::Mat>  faceDatabaseFeature(const string path, Net detectNet, Net net, Ptr< cv::face::Facemark> facemark)
{

    std::vector<cv::String> result;
    cv::glob(path, result,true);

    std::map<std::string,cv::Mat> feature_v;
    for (size_t i = 0; i < result.size(); i++)
    {
        cv::Mat img = cv::imread(result[i]);
        if (img.empty())
        {
            cerr<<"cann't open the image";
            break;
        }

        if (img.channels() == 4)
            cvtColor(img, img, cv::COLOR_BGRA2BGR);

        std::vector<Rect> faces = detectFaces( detectNet, img);

//        std::vector< std::vector<Point2f> > landmarks;
//
//        facemark->fit(img,faces, landmarks);
//
//        cv::Mat facepart = face_alignment(img(faces[0]), landmarks[0][36], landmarks[0][45], faces[0]);
//        imshow("", facepart);
//        waitKey(100);

        cv::Mat facepart = img(faces[0]);

        //虽然这里只取最大的一个faces[0]，但是作为可扩展的,保留
        cv::Mat feature= face2vec(net, facepart,1.0/255,96,96,cv::Scalar(0,0,0));
        feature_v.emplace(make_pair(getShortName(result[i]),feature));
    }
    return feature_v;
}


void facerecognition()
{
    //加载人脸检测模型
    std::string modelConfiguration = "/home/z/data/OpenCVData/deploy_lowres.prototxt";
    std::string modelBinary = "/home/z/data/OpenCVData/res10_300x300_ssd_iter_140000_fp16.caffemodel";
    //初始化网络
    Net detectNet = readNetFromCaffe(modelConfiguration, modelBinary);
    CV_Assert(!detectNet.empty());

    // Create an instance of Facemark and Load landmark detector
    Ptr< cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel("/home/z/data/OpenCVData/lbfmodel.yaml");

    //加载人脸识别模型
    std::string recongnitionModel = "/home/z/data/OpenCVData/openface.nn4.small2.v1.t7";
    Net recoNet = readNetFromTorch(recongnitionModel);
    CV_Assert(!recoNet.empty());

    //读取数据库图片路径
    cv::String dataBasePath="/home/z/data/secret/FaceRegister/FaceRegister";
    std::map<std::string,cv::Mat>  dataBaseF = faceDatabaseFeature(dataBasePath,detectNet,recoNet, facemark);
    CV_Assert(dataBaseF.size() > 0);


    //读取待识别图片路径
    cv::String pattern="/home/z/data/secret/37personimages";
    std::vector<cv::String> result;
    cv::glob(pattern, result,true);
    CV_Assert(result.size() > 0);

    //统计正确数量
    size_t count = 0, allNum = 0;

    //定义寻找最大值的变量
    double bestMatchScore = 0.5;
    string bestMatchName  = "unknown";
    for (size_t j = 0; j < result.size(); ++j)
    {

        cv::Mat img = imread(result[j]);
        if (img.empty())
        {
            cerr<<"cann't open the image";
            continue;
        }

        if (img.channels() == 4)
            cvtColor(img, img, cv::COLOR_BGRA2BGR);

        std::vector<Rect> faces = detectFaces(detectNet, img);

        if (faces.empty() || 0 > faces[0].x || 0 > faces[0].width || faces[0].x + faces[0].width > img.cols || 0 > faces[0].y || 0 > faces[0].height || faces[0].y + faces[0].height > img.rows)
        {
            cerr<<"cann't locate the face"<<std::endl;
            continue;
        }
//        std::cout<<"faces"<<faces[0].x<<" "<<faces[0].y<<"  "<<faces[0].width<<"  "<<faces[0].height<<std::endl;

//        std::vector< std::vector<Point2f> > landmarks;
//
//        facemark->fit(img,faces, landmarks);
//
//        cv::Mat facepart = face_alignment(img(faces[0]), landmarks[0][36], landmarks[0][45], faces[0]);
        cv::Mat facepart = img(faces[0]);

        //虽然这里只取最大的一个faces[0]，但是作为可扩展的,保留
        cv::Mat feature = face2vec(recoNet, facepart,1.0/255,96,96,cv::Scalar(0,0,0));
//        std::cout<<"feature is: "<<feature<<std::endl;
//        std::cout<<"feature size is:"<<feature.size<<std::endl;

        for (auto fe : dataBaseF)
        {
            double score = feature.dot(fe.second);//有问题
            if (score > bestMatchScore)
            {
                bestMatchScore = score;
                bestMatchName  = getShortName(fe.first);
            }
        }


        if (result[j].find(bestMatchName) > 0 && result[j].find(bestMatchName) < result[j].length() )
        {
            count++;
            std::cout<<result[j].substr(pattern.length()+1, result[j].find_last_of('/')-pattern.length()-2);
            std::cout<<"  max score is "<< bestMatchScore<<"   max name is: "<<bestMatchName<<std::endl;
        }
        allNum++;

        bestMatchScore = 0 ;
        bestMatchName  = "unknow";



    }

    double accuracy = count*1.0/allNum;
    std::cout<<count<<"   "<<allNum<<"   "<<accuracy<<std::endl;

}

