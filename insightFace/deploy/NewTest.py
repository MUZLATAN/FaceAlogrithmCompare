import face_model
import argparse
import cv2
import os
import numpy as np
import time


def face_distance(data_features, feature_to_compare):
    if len(data_features) == 0:
        return np.empty((0))
    face_dist_value = np.linalg.norm(data_features - feature_to_compare, axis=1)
    # print('[Face Services | face_distance] Distance between two faces is {}'.format(face_dist_value))
    return face_dist_value


class TestClass:
    Model = None
    registerdirs = "/home/z/data/secret/FaceRegister/FaceRegister"
    # recognitiondirs = "/home/z/data/secret/37personimages/p7_Jiangxianjuan"
    recognitiondirs = "/home/z/data/secret/37personimages"
    register_name = []
    no_detect_num = 0
    detect_num = 0
    recognize_rightly_num = 0
    recognize_wrongly_num = 0
    refuse_num = 0

    names = []
    face_features = []

    def __init__(self, model):
        self.Model = model

    def add_face(self, name, face_feature):
        self.names.append(name)
        self.face_features.append(face_feature)

    def registerFace(self, name):
        print(name)
        img = cv2.imread(self.registerdirs+"/"+name)
        img = model.get_input(img)
        f1 = model.get_feature(img)
        self.add_face(name, f1)

    def get_similar_faces(self, name):
        img = cv2.imread(name)
        img = model.get_input(img)
        if img is None:
            self.no_detect_num += 1
            return -1, -1

        face_feature = model.get_feature(img)

        if len(self.names) == 0:
            return -1, -1
        distant_value = face_distance(self.face_features, face_feature)
        # print(distant_value)
        min_value, idx = min(distant_value), np.argmin(distant_value)

        return min_value, idx


def get_file_path(root_path, file_list, dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)

def get_short_name(name):
    # str = "/home/z/data/secret/37personimages/p40_Wangwenhe/1571278883-6e4fa937-2d0e-446e-b149-a5c84956b43e.jpg"
    return name.split('/')[6].split('_')[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')

    parser.add_argument('--model', default='/home/z/PYWorkspace/insightFace/models/model-r100-ii/model,0', help='path to load model.')

    parser.add_argument('--ga-model', default='/home/z/PYWorkspace/insightFace/models/gamodel-r50/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='cpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    args = parser.parse_args()
    print(args)
    model = face_model.FaceModel(args)

    testcl = TestClass(model)

    file_list = []
    dir_list = []
    all_num = 0

    get_file_path(testcl.recognitiondirs, file_list, dir_list)

    re_dirs = os.listdir(testcl.registerdirs)
    for file in re_dirs:
        testcl.registerFace(file)

    # cap = cv2.VideoCapture("/home/z/Desktop/unnormal.webm")
    # while True:
    #     ret, img = cap.read()
    #     cv2.imshow("", img)
    #     cv2.waitKey(10)
    #     model.get_input(img)
    # cap.release()
    #
    # testcl.get_similar_faces("/home/z/data/secret/37personimages/p7_Jiangxianjuan/1571382798-3eccbda2-be6e-46d0-9011-ac8e4767bd3b.jpg")


    for file in file_list:
        all_num += 1
        # print(file)
        min_value, idx =testcl.get_similar_faces(file)
        if min_value != -1 & idx != -1:
            if min_value > 1:
                testcl.refuse_num += 1
                continue

            match_name = testcl.names[idx]
            short_name = get_short_name(file)

            if match_name.find(short_name) != -1:
                testcl.recognize_rightly_num += 1
                print(short_name, match_name)
            # else:
            #     print(short_name, match_name)

            if testcl.recognize_rightly_num % 10 == 0:
                print(testcl.recognize_rightly_num)

    print("all num is:", all_num, "no detect face num is: ", testcl.no_detect_num, "refuse num is: ", testcl.refuse_num, "number of recognize rightly", testcl.recognize_rightly_num)

    print(model.detect_face_time, model.get_feature_time)
