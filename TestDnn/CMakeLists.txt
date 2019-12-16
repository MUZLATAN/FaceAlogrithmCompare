cmake_minimum_required(VERSION 3.14)
project(TestDlib)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
add_compile_options(-Wno-deprecated-declarations)
add_compile_options(-DECVAPI_EXPORTS)
add_compile_options(-fPIC -Wall -Wno-unused-function)
add_compile_options(-msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mssse3 -mavx -mf16c -mavx2 -mfma)

# for lib root
set(CMAKE_MODULE_PATH "/opt/Devs")

##for ecv
find_package(ECV5 REQUIRED)
include_directories(${ECV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${ECV_LIBS})

# for opencv root
find_package(OpenCV REQUIRED core  highgui imgproc imgcodecs dnn face)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV_INCLUDE_DIRS:  ${OpenCV_INCLUDE_DIRS}")
set(LIBS ${LIBS} ${OpenCV_LIBS})
message(${OpenCV_LIBS})

#depend
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

# protobuf
#for protobuf
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Protobuf_LIBRARIES})

#for lib tbb
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
set(LIBS ${LIBS} ${TBB_LIBS})


#add_subdirectory(TEST)

add_executable(TestDlib main.cpp TestFunc.cpp TestFunc.h)
#/opt/Devs/dlib/lib/libdlib.a
target_link_libraries(TestDlib ${LIBS} )