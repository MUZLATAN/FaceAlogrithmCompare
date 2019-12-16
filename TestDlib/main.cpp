#include <iostream>
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_loader/load_image.h"
#include "dlib/image_saver/save_jpeg.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include "dlib/opencv.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace dlib;
using namespace cv;
std::time_t get_timestamp()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    std::time_t timestamp = tmp.count();
    return timestamp;
}
int main() {
    try {

        frontal_face_detector detector = get_frontal_face_detector();

        //use dnn detect the face
        shape_predictor sp;
        deserialize("/opt/Devs/dlib/data/shape_predictor_68_face_landmarks.dat")>>sp;


        std::string name = "/home/z/data/myface/112.jpg";
        array2d<unsigned char> img;
        load_image(img, name);


        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces it can find in the image.
        std::vector<dlib::rectangle> dets;
        dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;


        // convert to opencv mat
        cv::Mat mat_img = toMat(img);
        cv::rectangle(mat_img, cv::Rect(dets.at(0).left(),dets.at(0).top(),dets.at(0).bottom()-dets.at(0).top(),dets.at(0).right()-dets.at(0).left()),cv::Scalar(0,255,0));


        std::vector<full_object_detection> shapes;
        for(unsigned long j = 0; j < dets.size(); ++j)
        {
            full_object_detection shape = sp(img, dets[j]);
            std::cout << "number of parts: " << shape.num_parts() << std::endl;
            std::cout << "pixel position of first part:  " << shape.part(0) << std::endl;
            std::cout << "pixel position of second part: " << shape.part(1) << std::endl;

            shapes.push_back(shape);
        }

        for (auto it : shapes)
        {
            for (size_t idx = 1; idx < it.num_parts(); ++idx)
            {
                std::cout<<it.part(idx)<<"  ";
                cv::Point first_p(it.part(idx-1).x(),it.part(idx-1).y());
                cv::Point second_p(it.part(idx).x(),it.part(idx).y());
                cv::line(mat_img,first_p,second_p,cv::Scalar(0,255,0));
            }
        }

        cv::imshow(" ",mat_img);
        cv::waitKey(0);

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }

    return 0;
}