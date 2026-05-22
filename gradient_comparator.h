#ifndef GRADIENT_COMPARATOR_H
#define GRADIENT_COMPARATOR_H

#include <opencv2/opencv.hpp>

class GradientComparator {
public:
    GradientComparator(int sobelKernel = 3);
    
   
    double compare(const cv::Mat &grayA, const cv::Mat &grayB, int nthreads = -1);
     
   
    
private:
    int sobelKernel_;
    
    cv::Mat compute_gradient_magnitude(const cv::Mat &gray) const;
};

#endif  