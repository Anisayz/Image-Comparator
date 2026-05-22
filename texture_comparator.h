#ifndef TEXTURE_COMPARATOR_H
#define TEXTURE_COMPARATOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class TextureComparator {
public:
    TextureComparator(int blocks = 16, int histBins = 64);
    

    double compare(const cv::Mat &grayA, const cv::Mat &grayB);
    
 
    
private:
    int blocks_;
    int histBins_;
    
 
    void compute_lbp(const cv::Mat &gray, cv::Mat &lbp) const;
    void compute_block_lbp_histogram(const cv::Mat &lbp, int bx, int by,
                                     std::vector<float> &hist) const;
    void compute_all_lbp_histograms(const cv::Mat &lbp,
                                    std::vector<std::vector<float>> &allHists) const;
};

#endif  