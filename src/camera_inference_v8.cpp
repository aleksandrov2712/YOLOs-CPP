#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <string>
#include <Windows.h>
#include <nlohmann/json.hpp>
#include <fstream>

#include "tools/BoundedThreadSafeQueue.hpp"
#include "tools/FFmpegPipe.hpp"
#include "tools/StatsDisplay.hpp"
#include "YOLO10.hpp"

// Получение пути к исполняемому файлу
static std::string getExecutablePath() {
    std::vector<char> pathBuf(static_cast<size_t>(MAX_PATH));
    DWORD length = GetModuleFileNameA(NULL, pathBuf.data(),
        static_cast<DWORD>(pathBuf.size()));
    if (length == 0) {
        throw std::runtime_error("Could not get executable path");
    }
    std::string path(pathBuf.data(), length);
    size_t lastSlash = path.find_last_of("\\/");
    return lastSlash != std::string::npos ? path.substr(0, lastSlash) : path;
}

// Загрузка конфигурации из JSON файла
nlohmann::json loadConfig(const std::string& configPath) {
    std::ifstream configFile(configPath);
    if (!configFile.is_open()) {
        throw std::runtime_error("Could not open config file: " + configPath);
    }
    nlohmann::json config;
    configFile >> config;
    return config;
}

int main() {
    try {
        // Load configuration from JSON file
        const std::string configPath = "config.json";
        nlohmann::json config = loadConfig(configPath);

        // Fetch configuration parameters
        const std::string exePath = getExecutablePath();
        const std::string ffmpegPath = exePath + config["ffmpeg_path"].get<std::string>();
        const std::string rtspUrl = config["rtsp_url"];
        const int width = config["width"];
        const int height = config["height"];
        const std::string modelPath = exePath + config["model_path"].get<std::string>();
        const std::string labelsPath = exePath + config["labels_path"].get<std::string>();

        std::cout << "Initializing YOLO detector..." << std::endl;
        YOLO10Detector detector(modelPath, labelsPath, false);

        std::cout << "Starting FFmpeg pipe..." << std::endl;
        FFmpegPipe ffmpeg(ffmpegPath, rtspUrl, width, height);

        cv::namedWindow("RTSP Stream", cv::WINDOW_NORMAL);
        cv::resizeWindow("RTSP Stream", width, height);

        const size_t maxQueueSize = config["max_queue_size"];
        BoundedThreadSafeQueue<cv::Mat> frameQueue(maxQueueSize);
        BoundedThreadSafeQueue<std::pair<cv::Mat, std::vector<Detection>>>
            processedQueue(maxQueueSize);
        std::atomic<bool> stopFlag(false);

        // Поток захвата кадров
        std::thread producer([&]() {
            cv::Mat frame;
            while (!stopFlag.load()) {
                if (ffmpeg.read(frame) && !frame.empty()) {
                    if (!frameQueue.enqueue(frame.clone())) {
                        break;
                    }
                }
                else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            frameQueue.set_finished();
            });

        // Поток обработки кадров
        std::thread consumer([&]() {
            const size_t batchSize = config["batch_size"];
            std::vector<cv::Mat> frameBatch;
            frameBatch.reserve(batchSize);

            cv::Mat frame;
            while (!stopFlag.load() && frameQueue.dequeue(frame)) {
                frameBatch.push_back(frame);

                if (frameBatch.size() >= batchSize) {
                    std::vector<std::vector<Detection>> batchDetections;
                    batchDetections.reserve(frameBatch.size());

                    // Используем size_t вместо int64_t
#pragma omp parallel for num_threads(4)
                    for (size_t i = 0; i < frameBatch.size(); i++) {
                        std::vector<Detection> detections = detector.detect(frameBatch[i]);
#pragma omp critical
                        batchDetections.push_back(std::move(detections));
                    }

                    for (size_t i = 0; i < frameBatch.size(); i++) {
                        if (!processedQueue.enqueue(std::make_pair(
                            std::move(frameBatch[i]),
                            std::move(batchDetections[i])))) {
                            break;
                        }
                    }

                    frameBatch.clear();
                }
            }
            processedQueue.set_finished();
            });

        // Инициализация статистики
        StatsDisplay stats;

        // Основной цикл отображения
        std::pair<cv::Mat, std::vector<Detection>> item;
        while (!stopFlag.load() && processedQueue.dequeue(item)) {
            // Подсчет людей с улучшенными критериями
            int personCount = std::count_if(
                item.second.begin(),
                item.second.end(),
                [](const Detection& det) {
                    const float minConfidence = 0.25f;
                    const int minWidth = 15;
                    const int minHeight = 30;
                    const float aspectRatioMin = 1.2f;
                    const float aspectRatioMax = 4.0f;

                    if (det.obj_id != 0) return false;
                    if (det.accuracy < minConfidence) return false;

                    int width = det.x2 - det.x1;
                    int height = det.y2 - det.y1;

                    if (width < minWidth || height < minHeight) return false;

                    float aspectRatio = static_cast<float>(height) / width;
                    if (aspectRatio < aspectRatioMin || aspectRatio > aspectRatioMax) return false;

                    return true;
                });

            // Обновление и отображение статистики
            stats.update(personCount);
            detector.drawBoundingBoxMask(item.first, item.second);
            stats.draw(item.first);

            cv::imshow("RTSP Stream", item.first);

            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q' || key == 27) {
                std::cout << "\nGracefully shutting down..." << std::endl;
                stopFlag.store(true);
                break;
            }
        }

        // Завершение работы
        stopFlag.store(true);
        producer.join();
        consumer.join();
        cv::destroyAllWindows();

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Critical error: " << e.what() << std::endl;
        return -1;
    }
}