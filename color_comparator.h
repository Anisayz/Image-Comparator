#ifndef COLOR_COMPARATOR_H
#define COLOR_COMPARATOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class ColorComparator {
public:
    ColorComparator(int blocks = 16, int histBins = 64);
    
    
    double compare(const cv::Mat &imgA, const cv::Mat &imgB);
    
 
    void setBlocks(int blocks) { blocks_ = blocks; }
    void setHistBins(int bins) { histBins_ = bins; }
    
  
    
private:
    int blocks_;
    int histBins_;
    std::vector<double> blockSimilarities_;
    
  
    void compute_block_histogram(const cv::Mat &img, int bx, int by,
                                 std::vector<std::vector<float>> &hist) const;
    void compute_all_histograms(const cv::Mat &img,
                                std::vector<std::vector<std::vector<float>>> &allHists) const;
};

#endif 