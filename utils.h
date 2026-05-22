#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

 
int get_thread_count();

 
void ensure_same_size(cv::Mat &A, cv::Mat &B);
cv::Mat convert_to_grayscale(const cv::Mat &color);

 
double bhattacharyya_coeff(const std::vector<float> &h1, const std::vector<float> &h2);

 
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
  
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        return static_cast<double>(duration.count());
    }
    
 
    double elapsed_seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start_);
        return duration.count();
    }
    
  
    double elapsed_microseconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return static_cast<double>(duration.count());
    }
    
 
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
   
    void print_elapsed(const std::string &label = "") const {
        if (!label.empty()) {
            std::cout << label << ": ";
        }
        std::cout << elapsed_ms() << " ms\n";
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

#endif 