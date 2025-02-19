#pragma once

#include <chrono>
#include <deque>
#include <string>
#include <iomanip>
#include <sstream>
#include <opencv2/opencv.hpp>

class StatsDisplay {
public:
    StatsDisplay()
        : startTime(std::chrono::steady_clock::now())
        , frameCount(0)
        , currentPersons(0)
        , maxPersons(0)
        , avgFps(0.0) {}

    void update(int personCount) {
        const auto now = std::chrono::steady_clock::now();
        ++frameCount;

        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            now - startTime).count();
        if (duration > 0) {
            avgFps = static_cast<double>(frameCount) / duration;
        }

        currentPersons = personCount;
        maxPersons = std::max(maxPersons, personCount);

        const size_t MAX_HISTORY = 300;
        if (personHistory.size() >= MAX_HISTORY) {
            personHistory.pop_front();
        }
        personHistory.push_back(personCount);
    }

    void draw(cv::Mat& frame) {
        const int padding = 10;
        const double fontScale = 0.6;
        const int thickness = 2;
        const int lineSpacing = 25;

        cv::Mat overlay;
        frame.copyTo(overlay);

        int maxTextWidth = 0;
        std::vector<std::string> stats = {
            "Persons detected: " + std::to_string(currentPersons),
            "Max persons: " + std::to_string(maxPersons),
            "FPS: " + formatFPS(avgFps),
            "Total frames: " + std::to_string(frameCount)
        };

        for (const auto& text : stats) {
            cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                fontScale, thickness, nullptr);
            maxTextWidth = std::max(maxTextWidth, textSize.width);
        }

        cv::Rect bg(padding, padding, maxTextWidth + 20,
            static_cast<int>(stats.size()) * lineSpacing + 15);

        cv::rectangle(overlay, bg, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);

        int y = padding + lineSpacing;
        for (const auto& text : stats) {
            drawText(frame, text, cv::Point(padding + 10, y),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness);
            y += lineSpacing;
        }
    }

private:
    std::chrono::steady_clock::time_point startTime;
    std::deque<int> personHistory;
    int frameCount;
    int currentPersons;
    int maxPersons;
    double avgFps;

    std::string formatFPS(double fps) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << fps;
        return oss.str();
    }

    void drawText(cv::Mat& frame, const std::string& text, const cv::Point& pos,
        int fontFace, double fontScale, int thickness) {
        cv::putText(frame, text, pos, fontFace, fontScale,
            cv::Scalar(0, 0, 0), thickness + 1, cv::LINE_AA);
        cv::putText(frame, text, pos, fontFace, fontScale,
            cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
};