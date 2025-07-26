#include "ImageHasher.h"

// Constructors
ImageHasher::ImageHasher() {
    image = cv::Mat();
    hash = 0;
}

ImageHasher::ImageHasher(fs::path p) {
    image = cv::imread(p.string());
    hash = 0;
}

ImageHasher::ImageHasher(string s) {
    image = cv::imread(s);
    hash = 0;
}

ImageHasher::ImageHasher(ImageHasher &i) {
    image = i.image.clone();
    hash = i.hash;
}


// Functions
void ImageHasher::preprocess_square_grayscale(const <#int#> DIM = 8) {
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

uint64_t ImageHasher::average_hash(const int DIM = 8) {
    // Expects img to be of type CV_8UC1
    double avg = cv::mean(image)[0];
    
    // Build 64-bit hash
    uint64_t hash = 0;
    for (int r = 0; r < DIM; r += 1) {
        for (int c = 0; c < DIM; c += 1) {
            hash <<= 1;
            if (image.at<uchar>(r, c) >= avg) {
                hash |= 1;
            }
        }
    }
    return hash;
}

// Operators
int ImageHasher::operator-(ImageHasher &i) {
    return this->hamming_distance(i);
}
