cmake_minimum_required(VERSION 3.0.0)
project(task)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# 寻找 OpenCV 库
find_package(OpenCV REQUIRED)

# 设置 Ceres Solver 头文件路径
include_directories("/home/daiwei/ceres-solver-1.14.0/include")

# 寻找 Ceres Solver 库
find_package(Ceres REQUIRED)
# 将所有源文件添加到变量 ALL_SRCS 中
aux_source_directory(./src ALL_SRCS)

# 添加包含路径
include_directories(include ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

# 创建可执行文件并链接库
add_executable(task main.cpp ${ALL_SRCS})
target_link_libraries(task ${OpenCV_LIBS} ${CERES_LIBRARIES})
