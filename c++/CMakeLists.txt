cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project( TrainSVM )

find_package( OpenCV HINTS "/usr/local/opt/opencv@3" )
set(CMAKE_CXX_FLAGS  "-lcblas -lclapack")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable( TrainSVM TrainSVM.cpp )
set_property(TARGET TrainSVM PROPERTY CXX_STANDARD 11)
set_property(TARGET TrainSVM PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries( TrainSVM ${OpenCV_LIBS} )

add_executable( TestSVM TestSVM.cpp )
set_property(TARGET TestSVM PROPERTY CXX_STANDARD 11)
set_property(TARGET TestSVM PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries( TestSVM ${OpenCV_LIBS} )
