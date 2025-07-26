#include "ImageHasher.h"

// Constructors
ImageHasher::ImageHasher() {
    image = cv::Mat();
    hash = 0;
}

ImageHasher::ImageHasher(fs::path p) {
    image = cv::imread(p.string());
    hash = 0;
    file = p;
}

ImageHasher::ImageHasher(string s) {
    image = cv::imread(s);
    hash = 0;
    file = fs::path(s);
}

ImageHasher::ImageHasher(ImageHasher &i) {
    image = i.image.clone();
    hash = i.hash;
    file = i.file;
}


// Functions
void ImageHasher::resize_grayscale(const int DIM) {
    // Resize image to square and convert to grayscale
    cv::Mat resized, gray;
    cv::resize(image, resized, cv::Size(DIM, DIM));
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    image = gray;
}

cv::Mat& ImageHasher::normalize_image() {
    // image is assigned a new object in new memory address
    if (image.type() == CV_32FC1) {
        cv::Mat out1, out2;
        cv::normalize(image, out1, 0.0, 1.0, cv::NORM_MINMAX);
        out1.convertTo(out2, CV_8U, 255.0);
        image = out2.clone();
    }
    return image;
}

int ImageHasher::hamming_distance(ImageHasher &i) {
    uint64_t diff = hash ^ i.hash;
    int count = 0;
    while(diff > 0) {
        count += diff & 1;
        diff >>= 1;
    }
    return count;
}

int ImageHasher::hamming_distance(uint64_t h) {
    uint64_t diff = hash ^ h;
    int count = 0;
    while(diff > 0) {
        count += diff & 1;
        diff >>= 1;
    }
    return count;
}

// Hashing algorithms
uint64_t ImageHasher::average_hash() {
    // Expects img to be of type CV_8UC1
    double avg = cv::mean(image)[0];
    
    // Build 64-bit hash
    uint64_t h = 0;
    for (int r = 0; r < 8; r += 1) {
        for (int c = 0; c < 8; c += 1) {
            h <<= 1;
            if (image.at<uchar>(r, c) >= avg) {
                h |= 1;
            }
        }
    }
    hash = h;
    return h;
}

// Hashing pipelines
pair<string, uint64_t> ImageHasher::average_hash_pipeline() {
    // Resize and convert to grayscale
    resize_grayscale_8();
    
    // Normalize if necessary
    normalize_image();
    
    // Return hash
    average_hash();
    return pair(file.string(), hash);
}

// Operators
int ImageHasher::operator-(ImageHasher &i) {
    return this->hamming_distance(i);
}
