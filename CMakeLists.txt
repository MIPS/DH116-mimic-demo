cmake_minimum_required(VERSION 3.28)
project(dh116_mimic_demo VERSION 1.0 LANGUAGES C CXX )
set(CMAKE_C_STANDARD 11) 
set(CMAKE_C_STANDARD_REQUIRED ON) 
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}") 
set(PROJECT_INCLUDE_DIR "${PROJECT_ROOT}/include") 
set(PROJECT_MODEL_DIR "${PROJECT_ROOT}/models")

# SOEM
set(SOEM_DIR "${PROJECT_ROOT}/third_party/SOEM" CACHE PATH "Path to SOEM source tree")
if(NOT EXISTS "${SOEM_DIR}/CMakeLists.txt") 
    message(FATAL_ERROR "SOEM source not found at: ${SOEM_DIR}\n" "Initialize the submodule with:\n" " git submodule update --init --recursive\n" "Alternatively, configure with -DSOEM_DIR=/path/to/SOEM" ) 
endif()
add_subdirectory("${SOEM_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/third_party/SOEM" EXCLUDE_FROM_ALL)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(soem PRIVATE _POSIX_C_SOURCE=200809L)
endif()

# Optional C++ camera + ONNX Runtime frontend
option(BUILD_HAND_CAM "Build the camera frontend (requires OpenCV and ONNX Runtime)" OFF)
option(HAND_CAM_HEADLESS "Build the camera frontend without the GUI preview (no OpenCV highgui)" OFF)
set(ONNXRUNTIME_DIR "${PROJECT_ROOT}/third_party/onnxruntime" CACHE PATH "Path to ONNX Runtime source tree")

# C EtherCAT teleoperation controller
add_executable(dh116_teleop src/dh116_teleop.c src/dh116_bus.c src/hand_math.c )
target_include_directories(dh116_teleop PRIVATE "${PROJECT_INCLUDE_DIR}")
target_link_libraries(dh116_teleop PRIVATE soem )
target_compile_options(dh116_teleop PRIVATE -Wall -Wextra -Wpedantic )
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(dh116_teleop PRIVATE _POSIX_C_SOURCE=200809L _DEFAULT_SOURCE)
    target_link_libraries(dh116_teleop PRIVATE m)
endif()

if(BUILD_HAND_CAM)
    set(_opencv_components core imgproc videoio imgcodecs)
    if(NOT HAND_CAM_HEADLESS)
        list(APPEND _opencv_components highgui)
    endif()
    find_package(OpenCV REQUIRED COMPONENTS ${_opencv_components})

    # Override these paths if ONNX Runtime is installed in a nonstandard location.
    set(ONNXRUNTIME_ROOT "" CACHE PATH "Optional ONNX Runtime installation prefix")
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_DIR}/include"
        PATH_SUFFIXES onnxruntime onnxruntime/core/session core/session
        REQUIRED
    )
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
        REQUIRED
    )

    add_executable(hand_cam src/hand_cam.cpp)
    target_include_directories(hand_cam PRIVATE
        "${PROJECT_INCLUDE_DIR}" "${OpenCV_INCLUDE_DIRS}" "${ONNXRUNTIME_INCLUDE_DIR}"
    )
    target_link_libraries(hand_cam PRIVATE ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}")
    # A static ONNX Runtime does not carry its system library dependencies with it.
    if(ONNXRUNTIME_LIBRARY MATCHES "\\.a$")
        find_package(Threads REQUIRED)
        target_link_libraries(hand_cam PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")
            target_link_libraries(hand_cam PRIVATE atomic)
        endif()
    endif()
    target_compile_options(hand_cam PRIVATE -Wall -Wextra -Wpedantic)
    if(HAND_CAM_HEADLESS)
        target_compile_definitions(hand_cam PRIVATE HAND_CAM_HEADLESS)
    endif()
endif()

# Optional: hardware retargeting test
option(BUILD_RETARGET_TEST "Build the hardware-dependent retargeting test" OFF )
if(BUILD_RETARGET_TEST) 
add_executable(retarget_test src/test/retarget_test.c src/dh116_bus.c )
target_include_directories(retarget_test PRIVATE
    "${PROJECT_INCLUDE_DIR}"
)
target_link_libraries(retarget_test PRIVATE
    soem
)
target_compile_options(retarget_test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(retarget_test PRIVATE _POSIX_C_SOURCE=200809L _DEFAULT_SOURCE)
endif()
endif()

# Diagnostics
message(STATUS "Project root: ${PROJECT_ROOT}")
message(STATUS "SOEM directory: ${SOEM_DIR}")
message(STATUS "ONNX Runtime directory: ${ONNXRUNTIME_DIR}")
message(STATUS "Build hand camera frontend: ${BUILD_HAND_CAM}")
if(BUILD_HAND_CAM)
    message(STATUS "Hand camera headless: ${HAND_CAM_HEADLESS}")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "ONNX Runtime header: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()
message(STATUS "Model directory: ${PROJECT_MODEL_DIR}")
message(STATUS "Build retarget test: ${BUILD_RETARGET_TEST}")