# 人脸算法测试



2019.12.12 created by zlatan (z)

本次测试主要测试:

1. Dlib人脸检测 c++
2. 基于SSD 框架的 ResNet-10  restnet face c++
3. 基于caffe 框架的MTCNN人脸检测  c++
4. 基于mxne框架的insightFace人脸识别模型  python
5. 基于tensorflow的insightface_TF人脸识别模型   python
6. 基于Troch的openface模型  c++

> 实验所用的机器 *`(Intel® Core™ i7-6700K CPU @ 4.00GHz × 8  GeForce GTX 1060 6GB/PCIe/SSE2)`*



## Dlib(C++)

使用的是Dlib自带d人脸检测,由于原始算法没有采用Dlib人脸识别,因此只测试了Dlib人脸检测

1. [Dlib](http://dlib.net/dnn_mmod_face_detection_ex.cpp.html)代码参考

|      | 数据库 | 像素大小 | 检测数 | 正确率 | 检测平均耗时 | landmark | 模式    |
| ---- | ------ | -------- | ------ | ------ | ------------ | -------- | ------- |
| cpu  | 1987   | 960x540  | 1336   | 67.23% | 43.3ms       | 68       | Release |

__注:__

1. 使用的Release 模式, Debug模式下,耗时很长
2. 测试代码见本文件夹TestDlib



## Restnet face   (C++ )

opencv 的sample使用的模型, Caffe框架,     SSD framework using ResNet-10  restnet face 

1. 到官网下载模型文件[deploy_lowres.prototxt](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)和download_weights.py的文件,运行得到res10_300x300_ssd_iter_140000_fp16.caffemodel模型
2. 下载[resnet_ssd_face.cpp](https://github.com/opencv/opencv/blob/1073175c77885c6954ebfd96cfdaa3dc15cbc46f/samples/dnn/resnet_ssd_face.cpp)文件,仿照上面即可运行
3.  关于如何训练对应的模型,详细情况见how_to_train_face_detector.txt

|      | 数据库 | 像素大小 | 检测数 | 正确率 | 检测平均耗时 | landmark | 模式    |
| ---- | ------ | -------- | ------ | ------ | ------------ | -------- | ------- |
| cpu  | 1987   | 960x540  | 1975   | 99.39% | 19.11ms      |          | Release |

__注:__

1. Release模式 可以使用opencl 加速
2. 测试代码见本文件夹TsetDnn

##  MTCNN (C++)

在960x544大小的图像上测试, 数据集大小为1987

1. github上[MTCNN](https://github.com/imistyrain/MTCNN)的源码, 
2. 下载Fast_MTCNN和对应的model文件夹里面的模型

|      | 数据库 | 像素大小 | 检测数 | 正确率 | 检测平均耗时 | landmark | 模式    |
| ---- | ------ | -------- | ------ | ------ | ------------ | -------- | ------- |
| cpu  | 1987   | 960x544  | 1739   | 87.51% | 50.3ms       | 5        | Release |

1. 其中有200多张侧脸,低头,遮挡的图片  landmark 5个
2. 采用的是5点仿射矫正方法
3. 测试代码见本文件夹Fast_MTCNN



## insightFace(python)

[insightface](https://github.com/deepinsight/insightface) 采用的是MTCNN进行人脸检测，使用的是caffe框架下训练的模型在*`$INSIGHTFACE/deploy/mtcnn-model/`*，使用的是*`cv2.warpAffine`*函数进行人脸对齐 ，使用*`model-r1000-ii`*模型进行人脸__识别__准备：

1. 准备好预训练模型
2. 将模型放到文件夹*`$INSIGHTFACE_ROOT/models/`*文件夹下面， 如*`$INSIGHTFACE_ROOT/models/model-r1000-ii`*
3. 执行*`$INSIGHTFACE/deploy/test.py`* 

对于单张人脸图片会被裁剪只*`(112x112)`* ,并且整个的对齐加获取特征时间22ms

测试人脸数据,使用python基于mxnet框架,对已有的数据集进行测试

|      | 数据库 | 数据库总数 | 识别数 | 未检测数 | 正确率         | 检测耗时   | 编码耗时    |
| ---- | ------ | ---------- | ------ | -------- | -------------- | ---------- | ----------- |
| cpu  | 39     | 15139      | 10203  | 278      | __*`68.66%`*__ | 149.0635ms | 763.379ms   |
| gpu  | 39     | 15139      | 10203  | 278      | __*`68.66%`*__ | 40.35655ms | 22.178555ms |

__注:__

1.  此数据集完全来自现场数据,质量效果较差,但是即使在这样的数据集上仍然有较好的效果,足以说明该算法的独到之处. 另外此数据集的像素普遍偏小,因此检测和编码(特征提取)速度很快
2.  没有在1987张图片的数据库人脸识别测试
3. 测试代码见insightFace 文件夹



## openFace (C++)

 使用Torch框架训练的模型,进行人脸识别 

1. 运行参考[js_face_recognition.html](https://github.com/opencv/opencv/blob/4.1.2/samples/dnn/js_face_recognition.html)  
2. 该模型在这两个数据集上面训练的  [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) and [FaceScrub](http://vintage.winklerbros.net/facescrub.html)

|      | 数据库 | 数据库总数 | 识别数 | 未检测数 | 正确率 | 识别耗时 | 模式    |
| ---- | ------ | ---------- | ------ | -------- | ------ | -------- | ------- |
| cpu  | 39     | 15139      | -      | -        | -      | -        | Release |

__注__

1. 识别人脸,效果不理想,OpenFace version 0.2.0 识别率在92.9% , 但是在我们的数据库上识别效果很差, 可能是没有对齐的原因
2. 测试代码见本文件夹下TestDnn

---

