cmake_minimum_required(VERSION 3.14)
project(Fast_MTCNN)

set(CMAKE_CXX_STANDARD 17)


###general compilation options
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
#set(CMAKE_SKIP_BUILD_RPATH TRUE)

add_compile_options(-Wno-deprecated-declarations)
add_compile_options(-DECVAPI_EXPORTS)
add_compile_options(-fPIC -Wall -Wno-unused-function)
add_compile_options(-msse2 -msse3 -msse4 -msse4.1 -msse4.2 -msse3 -mavx -mavx2 -mf16c -mfma)


#for module path
set(CMAKE_MODULE_PATH /home/z/Software/devs_2019.11.25/Devs)

#for ecv
find_package(ECV5 REQUIRED)
include_directories(${ECV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${ECV_LIBS})

# for opencv
find_package(OpenCV REQUIRED core highgui imgproc  dnn)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBS})

# for opencv dep lib
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

# for folly
find_package(Folly REQUIRED)
include_directories(${Folly_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Folly_LIBS})

# for folly 3rd
set(LIBS ${LIBS} double-conversion iberty event unwind ssl crypto)

# for boost
find_package(Boost REQUIRED filesystem regex thread program_options context system )
include_directories(${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

#for protobuf
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Protobuf_LIBRARIES})

#for lib tbb
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
set(LIBS ${LIBS} ${TBB_LIBS})

# for glog gflags
set(LIBS ${LIBS} glog gflags)

# for gtest
find_package(GTEST REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GTEST_LIBRARIES})

# for pthread
find_package (Threads REQUIRED)
set(LIBS ${LIBS} Threads::Threads)


add_executable(Fast_MTCNN mtcnn_opencv.cpp MTCNN.cpp MTCNN.h)


target_link_libraries(Fast_MTCNN ${LIBS})