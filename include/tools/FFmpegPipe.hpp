#pragma once

#include <string>
#include <iostream>
#include <Windows.h>
#include <io.h>
#include <fcntl.h>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

class FFmpegPipe {
public:
    FFmpegPipe(const std::string& ffmpegPath, const std::string& rtspUrl,
        int width = 640, int height = 480)
        : width(width), height(height),
        frameSize(static_cast<size_t>(width)* static_cast<size_t>(height) * 3),
        reconnectAttempts(0),
        maxReconnectAttempts(5) {

        validatePaths(ffmpegPath);
        command = createCommand(ffmpegPath, rtspUrl, width, height);
        startProcess();
    }

    ~FFmpegPipe() {
        if (pipe) {
            _pclose(pipe);
        }
    }

    bool read(cv::Mat& frame) {
        if (!pipe) {
            handleReconnection();
            return false;
        }

        std::vector<unsigned char> buffer(frameSize);
        size_t bytesRead = fread(buffer.data(), 1, frameSize, pipe);

        if (bytesRead != frameSize) {
            handleReadError(bytesRead);
            return false;
        }

        try {
            cv::Mat rawData(1, static_cast<int>(frameSize), CV_8UC1, buffer.data());
            cv::Mat decodedFrame = cv::Mat(height, width, CV_8UC3, rawData.data).clone();

            if (decodedFrame.empty()) {
                std::cerr << "Warning: Decoded frame is empty" << std::endl;
                return false;
            }

            frame = decodedFrame;
            reconnectAttempts = 0;
            return true;
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV error during frame decoding: " << e.what() << std::endl;
            return false;
        }
    }

    bool isOpened() const { return pipe != nullptr; }

private:
    FILE* pipe = nullptr;
    int width;
    int height;
    size_t frameSize;
    std::string command;
    int reconnectAttempts;
    const int maxReconnectAttempts;
    static constexpr int reconnectDelayMs = 2000;

    void validatePaths(const std::string& ffmpegPath) {
        if (GetFileAttributesA(ffmpegPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
            throw std::runtime_error("FFmpeg executable not found at: " + ffmpegPath);
        }
    }

    static std::string createCommand(const std::string& ffmpegPath,
        const std::string& rtspUrl,
        int width, int height) {
        std::ostringstream cmd;
        std::string fixedPath = ffmpegPath;
        std::replace(fixedPath.begin(), fixedPath.end(), '\\', '/');

        cmd << "\"\"" << fixedPath << "\" "
            << "-rtsp_transport tcp "
            << "-rtsp_flags prefer_tcp "
            << "-timeout 5000000 "
            << "-fflags nobuffer+discardcorrupt+fastseek "
            << "-flags low_delay "
            << "-probesize 32 "
            << "-analyzeduration 0 "
            << "-i \"" << rtspUrl << "\" "
            << "-f rawvideo "
            << "-pix_fmt bgr24 "
            << "-vsync 0 "
            << "-vf \"scale=" << width << ":" << height << ",fps=15\" "
            << "-max_delay 500000 "
            << "-reorder_queue_size 0 "
            << "-preset ultrafast "
            << "-tune zerolatency "
            << "-an -sn "
            << "-threads 4 "
            << "-loglevel warning "
            << "-\"";

        return cmd.str();
    }

    void startProcess() {
        pipe = _popen(command.c_str(), "rb");
        if (!pipe) {
            DWORD error = GetLastError();
            throw std::runtime_error("Failed to start FFmpeg process. Error code: " +
                std::to_string(error));
        }

        int fd = _fileno(pipe);
        if (fd != -1) {
            if (_setmode(fd, _O_BINARY) == -1) {
                std::cerr << "Warning: Failed to set binary mode. Error: " <<
                    errno << std::endl;
            }
        }
    }

    void handleReadError(size_t bytesRead) {
        static int consecutiveErrors = 0;

        if (bytesRead == 0) {
            consecutiveErrors++;
            if (consecutiveErrors > 5) {
                if (pipe) {
                    _pclose(pipe);
                    pipe = nullptr;
                }
                std::cerr << "Too many consecutive errors, restarting pipe..." << std::endl;
                try {
                    startProcess();
                    consecutiveErrors = 0;
                }
                catch (const std::exception& e) {
                    std::cerr << "Failed to restart pipe: " << e.what() << std::endl;
                }
            }
        }
        else {
            consecutiveErrors = 0;
        }

        if (pipe == nullptr) {
            std::cerr << "Pipe is null" << std::endl;
            return;
        }

        std::cerr << "Read error: Got " << bytesRead << " bytes instead of " <<
            frameSize << std::endl;

        if (feof(pipe)) {
            std::cerr << "End of pipe reached, attempting to restart..." << std::endl;
            _pclose(pipe);
            pipe = nullptr;
            try {
                startProcess();
            }
            catch (const std::exception& e) {
                std::cerr << "Failed to restart pipe: " << e.what() << std::endl;
            }
        }
        else if (ferror(pipe)) {
            std::cerr << "Pipe error occurred" << std::endl;
            clearerr(pipe);
        }
    }

    void handleReconnection() {
        if (reconnectAttempts < maxReconnectAttempts) {
            std::cout << "Attempting to reconnect (" <<
                reconnectAttempts + 1 << "/" <<
                maxReconnectAttempts << ")..." << std::endl;

            if (pipe) {
                _pclose(pipe);
                pipe = nullptr;
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(reconnectDelayMs));

            try {
                startProcess();
                reconnectAttempts++;
            }
            catch (const std::exception& e) {
                std::cerr << "Reconnection failed: " << e.what() << std::endl;
            }
        }
    }
};