cmake_minimum_required(VERSION 3.10)

project(dso-ssl VERSION 1.0)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(TBB REQUIRED)
find_package(Pangolin REQUIRED)
find_package(SLAMViewer REQUIRED)
find_package(g2o REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tabulate REQUIRED)

set(BUILD_TESTS ON)

if(BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif()

# add dso_ssl library
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cc)
add_library(dso-ssl SHARED ${SRC_FILES})
target_link_libraries(dso-ssl PUBLIC ${OpenCV_LIBS} Eigen3::Eigen TBB::tbb
                                     yaml-cpp tabulate::tabulate g2o::core g2o::solver_eigen)
target_include_directories(
    dso-ssl
    PUBLIC ${OpenCV_INCLUDE_DIRS}
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)

message(STATUS ${SRC_FILES})
