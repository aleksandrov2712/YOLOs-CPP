#pragma once

// ===================================
// Single YOLOv10 Detector Header File
// ===================================
//
// This header defines the YOLO10Detector class for performing object detection using the YOLOv10 model.
// It includes necessary libraries, utility structures, and helper functions to facilitate model inference
// and result postprocessing.
//
// Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
// Date: 29.09.2024
//
// ================================

/**
 * @file YOLO10Detector.hpp
 * @brief Header file for the YOLO10Detector class, responsible for object detection
 *        using the YOLOv10 model with optimized performance for minimal latency.
 */

 // Include necessary ONNX Runtime and OpenCV headers
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Include standard libraries for various utilities
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <random>
#include <memory>
#include <thread>
#include <numeric>  // For std::accumulate
#include <cmath>    // For std::round

// Include debug and custom ScopedTimer tools for performance measurement
#include "tools/Debug.hpp"
#include "tools/ScopedTimer.hpp"

/**
 * @brief Confidence threshold for filtering detections.
 */
const float CONFIDENCE_THRESHOLD = 0.6f;

/**
 * @brief Struct to represent a single detection with bounding box.
 */
struct Detection {
    int x1, x2, y1, y2; // Coordinates of the bounding box
    int obj_id;         // Object class ID
    float accuracy;     // Confidence score of the detection

    // Constructor to initialize all members
    Detection(int x1, int x2, int y1, int y2, int obj_id, float accuracy)
        : x1(x1), x2(x2), y1(y1), y2(y2), obj_id(obj_id), accuracy(accuracy) {}
};

/**
 * @namespace utils
 * @brief Namespace containing utility functions for the YOLO10Detector.
 */
namespace utils {
    /**
     * @brief Loads class names from a given file path.
     *
     * @param path Path to the file containing class names.
     * @return std::vector<std::string> Vector of class names.
     */
    std::vector<std::string> getClassNames(const std::string& path) {
        std::vector<std::string> classNames;
        std::ifstream infile(path);

        if (infile) {
            std::string line;
            while (getline(infile, line)) {
                // Remove carriage return if present (for Windows compatibility)
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                classNames.emplace_back(line);
            }
        }
        else {
            std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
        }

        DEBUG_PRINT("Loaded class names from " + path);
        return classNames;
    }

    /**
     * @brief Computes the product of elements in a vector.
     *
     * @param vector Vector of integers.
     * @return size_t Product of all elements.
     */
    size_t vectorProduct(const std::vector<int64_t>& vector) {
        return std::accumulate(vector.begin(), vector.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    }

    /**
     * @brief Resizes an image with letterboxing to maintain aspect ratio.
     *
     * @param image Input image.
     * @param outImage Output resized and padded image.
     * @param newShape Desired output size.
     * @param color Padding color (default is gray).
     * @param auto_ Automatically adjust padding to be multiple of stride.
     * @param scaleFill Whether to scale to fill the new shape without keeping aspect ratio.
     * @param scaleUp Whether to allow scaling up of the image.
     * @param stride Stride size for padding alignment.
     */
    inline void letterBox(const cv::Mat& image, cv::Mat& outImage,
        const cv::Size& newShape,
        const cv::Scalar& color = cv::Scalar(114, 114, 114),
        bool auto_ = true,
        bool scaleFill = false,
        bool scaleUp = true,
        int stride = 32) {

        float ratio = std::min(static_cast<float>(newShape.height) / static_cast<float>(image.rows),
            static_cast<float>(newShape.width) / static_cast<float>(image.cols));

        if (!scaleUp) {
            ratio = std::min(ratio, 1.0f);
        }

        int newUnpadW = static_cast<int>(std::round(static_cast<float>(image.cols) * ratio));
        int newUnpadH = static_cast<int>(std::round(static_cast<float>(image.rows) * ratio));

        int dw = newShape.width - newUnpadW;
        int dh = newShape.height - newUnpadH;

        if (auto_) {
            dw = dw % stride;
            dh = dh % stride;
        }
        else if (scaleFill) {
            newUnpadW = newShape.width;
            newUnpadH = newShape.height;
            ratio = std::min(static_cast<float>(newShape.width) / static_cast<float>(image.cols),
                static_cast<float>(newShape.height) / static_cast<float>(image.rows));
            dw = 0;
            dh = 0;
        }

        int padLeft = dw / 2;
        int padRight = dw - padLeft;
        int padTop = dh / 2;
        int padBottom = dh - padTop;

        if (image.cols != newUnpadW || image.rows != newUnpadH) {
            cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
        }
        else {
            outImage = image.clone();
        }

        cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight,
            cv::BORDER_CONSTANT, color);
    }

    /**
     * @brief Scales detection coordinates back to the original image size.
     */
    void scaleResultCoordsToOriginal(const cv::Size& imageShape, Detection& bbox,
        const cv::Size& imageOriginalShape) {
        float gain = std::min(static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width),
            static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height));

        int padX = static_cast<int>((imageShape.width - imageOriginalShape.width * gain) / 2.0f);
        int padY = static_cast<int>((imageShape.height - imageOriginalShape.height * gain) / 2.0f);

        bbox.x1 = static_cast<int>((static_cast<float>(bbox.x1) - static_cast<float>(padX)) / gain);
        bbox.y1 = static_cast<int>((static_cast<float>(bbox.y1) - static_cast<float>(padY)) / gain);
        bbox.x2 = static_cast<int>((static_cast<float>(bbox.x2) - static_cast<float>(padX)) / gain);
        bbox.y2 = static_cast<int>((static_cast<float>(bbox.y2) - static_cast<float>(padY)) / gain);

        // Clip to image bounds
        bbox.x1 = std::max(0, std::min(bbox.x1, imageOriginalShape.width - 1));
        bbox.y1 = std::max(0, std::min(bbox.y1, imageOriginalShape.height - 1));
        bbox.x2 = std::max(0, std::min(bbox.x2, imageOriginalShape.width - 1));
        bbox.y2 = std::max(0, std::min(bbox.y2, imageOriginalShape.height - 1));
    }

    /**
     * @brief Draws bounding boxes and labels on the image based on detections.
     */
    inline void drawBoundingBox(cv::Mat& image, const std::vector<Detection>& detectionVector,
        const std::vector<std::string>& classNames,
        const std::vector<cv::Scalar>& colors) {
        for (const auto& detection : detectionVector) {
            if (detection.accuracy <= CONFIDENCE_THRESHOLD) continue;
            if (detection.obj_id < 0 || static_cast<size_t>(detection.obj_id) >= classNames.size()) continue;

            const cv::Scalar& color = colors[detection.obj_id % colors.size()];

            cv::rectangle(image, cv::Point(detection.x1, detection.y1),
                cv::Point(detection.x2, detection.y2), color, 2, cv::LINE_AA);

            const std::string& label = classNames[detection.obj_id];
            std::string confidenceText = std::to_string(static_cast<int>(detection.accuracy * 100)) + "%";

            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;
            int baseline = 0;

            cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
            cv::Size confidenceSize = cv::getTextSize(confidenceText, fontFace, fontScale, thickness, &baseline);

            cv::Point textOrg(detection.x1, std::max(detection.y1 - 10, textSize.height));
            cv::Point confidenceOrg(detection.x1,
                std::min(detection.y2 + confidenceSize.height + 10, image.rows - 1));

            // Draw background rectangles and text
            cv::rectangle(image, textOrg + cv::Point(0, baseline),
                textOrg + cv::Point(textSize.width, -textSize.height),
                color, cv::FILLED);
            cv::putText(image, label, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 0),
                thickness, cv::LINE_AA);

            cv::rectangle(image, confidenceOrg + cv::Point(0, baseline),
                confidenceOrg + cv::Point(confidenceSize.width, -confidenceSize.height),
                color, cv::FILLED);
            cv::putText(image, confidenceText, confidenceOrg, fontFace, fontScale,
                cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
        }
    }

    /**
     * @brief Generates a vector of colors for each class name.
     */
    inline const std::vector<cv::Scalar>& generateColors(const std::vector<std::string>& classNames,
        int seed = 42) {
        static std::unordered_map<size_t, std::vector<cv::Scalar>> colorCache;

        size_t hashKey = 0;
        for (const auto& name : classNames) {
            hashKey ^= std::hash<std::string>{}(name)+0x9e3779b9 + (hashKey << 6) + (hashKey >> 2);
        }

        auto it = colorCache.find(hashKey);
        if (it != colorCache.end()) {
            return it->second;
        }

        std::vector<cv::Scalar> colors;
        colors.reserve(classNames.size());

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> uni(0, 255);

        for (size_t i = 0; i < classNames.size(); ++i) {
            colors.emplace_back(cv::Scalar(uni(rng), uni(rng), uni(rng)));
        }

        colorCache[hashKey] = colors;
        return colorCache[hashKey];
    }

    /**
     * @brief Draws bounding boxes with semi-transparent masks.
     */
    inline void drawBoundingBoxMask(cv::Mat& image, const std::vector<Detection>& detections,
        const std::vector<std::string>& classNames,
        const std::vector<cv::Scalar>& classColors,
        float maskAlpha) {
        if (image.empty()) {
            std::cerr << "ERROR: Empty image provided to drawBoundingBoxMask." << std::endl;
            return;
        }

        cv::Mat maskImage(image.size(), image.type(), cv::Scalar::all(0));
        const double fontSize = std::min(image.rows, image.cols) * 0.0006;
        const int textThickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.001));

        std::vector<const Detection*> validDetections;
        validDetections.reserve(detections.size());

        for (const auto& detection : detections) {
            if (detection.accuracy > CONFIDENCE_THRESHOLD &&
                detection.obj_id >= 0 &&
                static_cast<size_t>(detection.obj_id) < classNames.size()) {
                validDetections.push_back(&detection);
            }
        }

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(validDetections.size()); ++i) {
            const Detection* detection = validDetections[i];
            cv::Rect box(detection->x1, detection->y1,
                detection->x2 - detection->x1,
                detection->y2 - detection->y1);
            const cv::Scalar& color = classColors[detection->obj_id];
            cv::rectangle(maskImage, box, color, cv::FILLED);
        }

        cv::addWeighted(maskImage, maskAlpha, image, 1.0, 0, image);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(validDetections.size()); ++i) {
            const Detection* detection = validDetections[i];
            cv::Rect box(detection->x1, detection->y1,
                detection->x2 - detection->x1,
                detection->y2 - detection->y1);
            const cv::Scalar& color = classColors[detection->obj_id];

            cv::rectangle(image, box, color, 2, cv::LINE_AA);

            char labelBuffer[256];
            std::snprintf(labelBuffer, sizeof(labelBuffer), "%s: %.0f%%",
                classNames[detection->obj_id].c_str(),
                detection->accuracy * 100.0f);
            std::string label(labelBuffer);

            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                fontSize, textThickness, &baseLine);

            int labelY = std::max(detection->y1, labelSize.height + 5);
            cv::Point labelTopLeft(detection->x1, labelY - labelSize.height - 5);
            cv::Point labelBottomRight(detection->x1 + labelSize.width + 5,
                labelY + baseLine - 5);

            cv::rectangle(image, labelTopLeft, labelBottomRight, color, cv::FILLED);
            cv::putText(image, label, cv::Point(detection->x1 + 2, labelY - 2),
                cv::FONT_HERSHEY_SIMPLEX, fontSize, cv::Scalar(255, 255, 255),
                textThickness, cv::LINE_AA);
        }
    }
} // namespace utils

/**
 * @brief Main detector class for YOLOv10.
 */
class YOLO10Detector {
public:
    /**
     * @brief Constructor to initialize the YOLO detector with model and label paths.
     */
    YOLO10Detector(const std::string& modelPath, const std::string& labelsPath, bool useGPU = false);

    /**
     * @brief Runs detection on the provided image.
     */
    std::vector<Detection> detect(const cv::Mat& image);

    /**
     * @brief Draws bounding boxes on the image based on detections.
     */
    void drawBoundingBox(cv::Mat& image, const std::vector<Detection>& detectionVector) const {
        utils::drawBoundingBox(image, detectionVector, classNames, classColors);
    }

    /**
     * @brief Draws bounding boxes with semi-transparent masks.
     */
    void drawBoundingBoxMask(cv::Mat& image, const std::vector<Detection>& detections,
        float maskAlpha = 0.4) const {
        utils::drawBoundingBoxMask(image, detections, classNames, classColors, maskAlpha);
    }

private:
    Ort::Env env{ nullptr };
    Ort::SessionOptions sessionOptions{ nullptr };
    Ort::Session session{ nullptr };
    bool isDynamicInputShape{};
    cv::Size inputImageShape;

    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char*> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char*> outputNames;

    size_t numInputNodes{}, numOutputNodes{};
    std::vector<std::string> classNames;
    std::vector<cv::Scalar> classColors;

    /**
     * @brief Preprocesses the input image for model inference.
     */
    cv::Mat preprocess(const cv::Mat& image, float*& blob,
        std::vector<int64_t>& inputTensorShape);

    /**
     * @brief Postprocesses the model output to extract detections.
     */
    std::vector<Detection> postprocess(const cv::Size& originalImageSize,
        const cv::Size& resizedImageShape,
        std::vector<Ort::Value>& outputTensors);
};

// Implementation of YOLO10Detector constructor
YOLO10Detector::YOLO10Detector(const std::string& modelPath,
    const std::string& labelsPath,
    bool useGPU) {
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    sessionOptions.SetIntraOpNumThreads(std::min(6,
        static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    std::wstring wModelPath(modelPath.begin(), modelPath.end());
    sessionOptions.SetOptimizedModelFilePath(wModelPath.c_str());
#else
    sessionOptions.SetOptimizedModelFilePath(modelPath.c_str());
#endif

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(),
        availableProviders.end(),
        "CUDAExecutionProvider");

    if (useGPU && cudaAvailable == availableProviders.end()) {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (useGPU && cudaAvailable != availableProviders.end()) {
        std::cout << "Inference device: GPU" << std::endl;
        OrtCUDAProviderOptions cudaOption;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else {
        std::cout << "Inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    session = Ort::Session(env, wModelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    isDynamicInputShape = inputTensorShape[2] == -1 && inputTensorShape[3] == -1;

    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    inputImageShape = cv::Size(static_cast<int>(inputTensorShape[3]),
        static_cast<int>(inputTensorShape[2]));

    numInputNodes = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    classNames = utils::getClassNames(labelsPath);
    classColors = utils::generateColors(classNames);

    std::cout << "Model loaded successfully with " << numInputNodes
        << " input nodes and " << numOutputNodes << " output nodes." << std::endl;
}

cv::Mat YOLO10Detector::preprocess(const cv::Mat& image, float*& blob,
    std::vector<int64_t>& inputTensorShape) {
    ScopedTimer timer("Preprocessing");

    cv::Mat resizedImage;
    utils::letterBox(image, resizedImage, inputImageShape, cv::Scalar(114, 114, 114),
        isDynamicInputShape, false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(resizedImage, CV_32FC3, 1.0f / 255.0f);

    size_t imageSize = static_cast<size_t>(resizedImage.total() * resizedImage.channels());
    blob = new float[imageSize];

    std::vector<cv::Mat> channels(resizedImage.channels());
    for (int i = 0; i < resizedImage.channels(); ++i) {
        channels[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1,
            blob + i * resizedImage.rows * resizedImage.cols);
    }
    cv::split(resizedImage, channels);

    DEBUG_PRINT("Preprocessing completed");
    return resizedImage;
}

std::vector<Detection> YOLO10Detector::postprocess(const cv::Size& originalImageSize,
    const cv::Size& resizedImageShape,
    std::vector<Ort::Value>& outputTensors) {
    ScopedTimer timer("Postprocessing");

    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int numDetections = static_cast<int>(outputShape[1]);

    std::vector<Detection> detectionVector;
    detectionVector.reserve(numDetections);

    DEBUG_PRINT("Processing " << numDetections << " detections");

    for (int i = 0; i < numDetections; i++) {
        const float confidence = rawOutput[i * 6 + 4];

        if (confidence < CONFIDENCE_THRESHOLD) continue;

        int x1 = static_cast<int>(std::round(rawOutput[i * 6 + 0]));
        int y1 = static_cast<int>(std::round(rawOutput[i * 6 + 1]));
        int x2 = static_cast<int>(std::round(rawOutput[i * 6 + 2]));
        int y2 = static_cast<int>(std::round(rawOutput[i * 6 + 3]));
        int classId = static_cast<int>(std::round(rawOutput[i * 6 + 5]));

        Detection det(x1, x2, y1, y2, classId, confidence);
        utils::scaleResultCoordsToOriginal(resizedImageShape, det, originalImageSize);
        detectionVector.push_back(det);
    }

    DEBUG_PRINT("Found " << detectionVector.size() << " valid detections");
    return detectionVector;
}

std::vector<Detection> YOLO10Detector::detect(const cv::Mat& image) {
    ScopedTimer timer("Overall detection");

    float* blobPtr = nullptr;
    std::vector<int64_t> inputTensorShape = { 1, 3,
                                             inputImageShape.height,
                                             inputImageShape.width };

    cv::Mat preprocessedImage = preprocess(image, blobPtr, inputTensorShape);
    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);
    delete[] blobPtr;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorSize,
        inputTensorShape.data(),
        inputTensorShape.size()
    );

    std::vector<Ort::Value> outputTensors = session.Run(
        Ort::RunOptions{ nullptr },
        inputNames.data(),
        &inputTensor,
        numInputNodes,
        outputNames.data(),
        numOutputNodes
    );

    cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]),
        static_cast<int>(inputTensorShape[2]));

    return postprocess(image.size(), resizedImageShape, outputTensors);
}