cmake_minimum_required(VERSION 3.10)
project(yolo_ort)

# ��������� ��������� C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ���������� ��������� �������������� MSVC
if(MSVC)
    add_compile_options(/wd26495 /wd26439)
    # ��������� runtime DLLs
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
    include(InstallRequiredSystemLibraries)
endif()

# ����������� vcpkg
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# �������� � ��������� ����� ��� OpenCV
if(NOT OpenCV_DIR)
    set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib" CACHE PATH "Path to OpenCV build directory")
endif()

# �������� ������������� ���������� OpenCV
if(NOT EXISTS ${OpenCV_DIR})
    message(FATAL_ERROR "OpenCV directory not found at: ${OpenCV_DIR}")
endif()

# ���� � �������� ������ OpenCV
get_filename_component(OPENCV_BIN_DIR "${OpenCV_DIR}/../bin" ABSOLUTE)

# �������� � ��������� ����� ��� ONNX Runtime
if(NOT ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime" CACHE PATH "Path to ONNX Runtime directory")
endif()

# �������� ������������� ���������� ONNX Runtime
if(NOT EXISTS ${ONNXRUNTIME_DIR})
    message(FATAL_ERROR "ONNX Runtime directory not found at: ${ONNXRUNTIME_DIR}")
endif()

# ����� OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please check the path: ${OpenCV_DIR}")
endif()

# ����� nlohmann_json
find_package(nlohmann_json REQUIRED)

# ���������� ���������� � ������������� �������
include_directories("include/")

# ���������� ������������ �����
add_executable(camera_inference src/camera_inference_v8.cpp)

# ��������� ��� camera_inference
target_include_directories(camera_inference 
    PRIVATE 
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)

# �������� ���������
target_link_libraries(camera_inference 
    PRIVATE 
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

if(MSVC)
    target_link_libraries(camera_inference 
        PRIVATE 
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    )
    
    # ����������� ��������� runtime DLLs
    add_custom_command(TARGET camera_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
            $<TARGET_FILE_DIR:camera_inference>
    )
    
    # ����������� ONNX Runtime DLL
    add_custom_command(TARGET camera_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:camera_inference>
    )
    
    # ����������� OpenCV DLLs
    file(GLOB OPENCV_DLLS 
        "${OPENCV_BIN_DIR}/opencv_world*.dll"
        "${OPENCV_BIN_DIR}/opencv_videoio_ffmpeg*.dll"
    )
    foreach(DLL ${OPENCV_DLLS})
        add_custom_command(TARGET camera_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL}
                $<TARGET_FILE_DIR:camera_inference>
        )
    endforeach()

    # �������� ����������
    add_custom_command(TARGET camera_inference PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:camera_inference>/models
            $<TARGET_FILE_DIR:camera_inference>/ffmpeg
    )

    # �������� ������� ������ �������
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/models/yolov8n-seg.onnx")
        message(FATAL_ERROR "YOLO model not found at: ${CMAKE_SOURCE_DIR}/models/yolov8n-seg.onnx")
    endif()

    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/models/coco.names")
        message(FATAL_ERROR "COCO names file not found at: ${CMAKE_SOURCE_DIR}/models/coco.names")
    endif()

    # ����������� FFmpeg
    add_custom_command(TARGET camera_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/ffmpeg"
            $<TARGET_FILE_DIR:camera_inference>/ffmpeg
    )

    # ����������� ������ �������
    add_custom_command(TARGET camera_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/models/yolo10n.onnx"
            "${CMAKE_SOURCE_DIR}/models/coco.names"
            $<TARGET_FILE_DIR:camera_inference>/models/
    )
    
    # ����������� config.json
    add_custom_command(TARGET camera_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/config.json"
            $<TARGET_FILE_DIR:camera_inference>
    )
endif()

# ����� ������������
message(STATUS "Build configuration:")
message(STATUS "  - OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  - OpenCV path: ${OpenCV_DIR}")
message(STATUS "  - ONNX Runtime path: ${ONNXRUNTIME_DIR}")
message(STATUS "  - C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  - Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  - System runtime libs: ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}")