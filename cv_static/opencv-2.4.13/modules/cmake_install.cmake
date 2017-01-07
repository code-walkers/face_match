# Install script for directory: /Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/androidcamera/.androidcamera/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/calib3d/.calib3d/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/contrib/.contrib/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/core/.core/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/dynamicuda/.dynamicuda/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/features2d/.features2d/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/flann/.flann/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/gpu/.gpu/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/highgui/.highgui/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/imgproc/.imgproc/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/java/.java/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/legacy/.legacy/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/ml/.ml/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/nonfree/.nonfree/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/objdetect/.objdetect/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/ocl/.ocl/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/photo/.photo/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/python/.python/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/stitching/.stitching/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/superres/.superres/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/ts/.ts/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/video/.video/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/videostab/.videostab/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/viz/.viz/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/world/.world/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/core/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/flann/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/imgproc/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/highgui/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/features2d/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/calib3d/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/ml/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/video/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/legacy/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/objdetect/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/photo/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/gpu/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/ocl/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/nonfree/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/contrib/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/stitching/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/superres/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/ts/cmake_install.cmake")
  include("/Users/arunporuri/Desktop/Face_match/cv_static/opencv-2.4.13/modules/videostab/cmake_install.cmake")

endif()

