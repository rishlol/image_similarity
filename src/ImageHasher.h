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
    inline fs::path getFile() {
        return file;
    }
    
    // Functions
    cv::Mat& normalize_image();
    void resize_grayscale(const int);
    int hamming_distance(uint64_t);

    // Inline functions
    inline void resize_grayscale_8() {
        resize_grayscale(8);
    };
    inline void resize_grayscale_32() {
        resize_grayscale(32);
    };
    inline int hamming_distance(ImageHasher& i) {
        return hamming_distance(i.hash);
    }
    
    // Hashing algorithms
    uint64_t average_hash();
    uint64_t perceptual_hash();
    
    // Hashing pipelines
    PipelineOut average_hash_pipeline();
    PipelineOut perceptual_hash_pipeline();
    
    // Operators
    int operator-(ImageHasher &i) {
        return this->hamming_distance(i);
    }
    int operator-(const uint64_t h) {
        return this->hamming_distance(h);
    }
};

#endif
