# cmake needs this line
cmake_minimum_required(VERSION 3.10)

# Define project name
project(vits)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "v")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

set(ESPEAK_NG_DIR ${PROJECT_SOURCE_DIR}/espeak-ng)
INCLUDE_DIRECTORIES(${ESPEAK_NG_DIR}/include/)
ADD_LIBRARY(espeak-ng SHARED IMPORTED)
set_property(TARGET espeak-ng PROPERTY IMPORTED_LOCATION ${ESPEAK_NG_DIR}/lib/libespeak-ng.so)


include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/main.cpp" "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.c" "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/*.hpp")


add_executable(${PROJECT_NAME} ${SOURCES})


target_include_directories(vits PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} )
target_link_libraries(vits PRIVATE ${ONNX_RUNTIME_LIB} espeak-ng)
