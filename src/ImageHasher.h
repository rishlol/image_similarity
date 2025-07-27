#ifndef IMAGEHASHER_H
#define IMAGEHASHER_H

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstdint>
#include <cmath>

using namespace std;
namespace fs = filesystem;

typedef pair<string, uint64_t> PipelineOut;
constexpr double PI = 3.1415926535897932384;

class ImageHasher {
private:
    cv::Mat image;
    uint64_t hash;
    fs::path file;
protected:
    void resize_grayscale(const int);
    inline double dct_constant(const int u) {
        return u == 0 ? 1.0 / sqrt(2) : 1.0;
    }
public:
    // Constructors
    ImageHasher();
    ImageHasher(fs::path);
    ImageHasher(string);
    ImageHasher(ImageHasher &);
    
    // Getters
    inline cv::Mat& getImage() {
        return image;
    }
    inline uint64_t getHash() {
        return hash;
    }
    
    // Functions
    inline void resize_grayscale_8() {
        resize_grayscale(8);
    };
    inline void resize_grayscale_32() {
        resize_grayscale(32);
    };
    cv::Mat& normalize_image();
    int hamming_distance(ImageHasher &);
    int hamming_distance(uint64_t);
    
    // Hashing algorithms
    uint64_t average_hash();
    uint64_t perceptual_hash();
    
    // Hashing pipelines
    pair<string, uint64_t> average_hash_pipeline();
    pair<string, uint64_t> perceptual_hash_pipeline();
    
    // Operators
    int operator-(ImageHasher &i) {
        return this->hamming_distance(i);
    }
    int operator-(const uint64_t h) {
        return this->hamming_distance(h);
    }
};

#endif
