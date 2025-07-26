#ifndef IMAGEHASHER_H
#define IMAGEHASHER_H

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstdint>

using namespace std;
namespace fs = filesystem;

class ImageHasher {
private:
    cv::Mat image;
    uint64_t hash;
public:
    ImageHasher();
    ImageHasher(fs::path);
    ImageHasher(string);
    ImageHasher(ImageHasher &);
    
    inline cv::Mat& getImage() {
        return image;
    }
    inline uint64_t getHash() {
        return hash;
    }
    
    void preprocess_square_grayscale(int);
    cv::Mat& normalize_image();
    int hamming_distance(ImageHasher &);
    uint64_t average_hash(int);
    
    int operator-(ImageHasher &);
};

#endif
