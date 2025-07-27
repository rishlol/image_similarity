#include "ImageHasher.h"
#include <vector>
#include <algorithm>

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

uint64_t ImageHasher::perceptual_hash() {
    cv::Mat dct(32, 32, CV_64F, cv::Scalar(0));
    
    // Populate dct matrix
    for (int r = 0; r < 32; r += 1) {
        for (int c = 0; c < 32; c += 1) {
            // Sum over original matrix
            for (int x = 0; x < 32; x += 1) {
                for (int y = 0; y < 32; y += 1) {
                    dct.at<double>(r, c) +=
                    .25 * dct_constant(r) * dct_constant(c) * image.at<uchar>(x, y) *
                    cos( ((2 * x + 1) * r * PI) / (2 * 32) ) * cos( ((2 * y + 1) * c * PI) / (2 * 32) );
                }
            }
        }
    }
    
    // Get top left 8x8 block
    vector<double> dct_values;
    for (int i = 0; i < 8; i += 1) {
        for (int j = 0; j < 8; j += 1) {
            dct_values.push_back(dct.at<double>(i, j));
        }
    }
    
    // Partial sort and store median (technically 33rd element)
    vector<double> sorted = dct_values;
    nth_element(sorted.begin(), sorted.begin() + 32, sorted.end());
    double median = sorted[32];
    
    // Iterate over dct values and create hash
    uint64_t h = 0;
    for (const double &val : dct_values) {
        h <<= 1;
        if(val > median) {
            h |= 1;
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

pair<string, uint64_t> ImageHasher::perceptual_hash_pipeline() {
    // Resize and convert to grayscale
    resize_grayscale_32();
    
    // Normalize if necessary
    normalize_image();
    
    // Return hash
    perceptual_hash();
    return pair(file.string(), hash);
}

// Operators
//int ImageHasher::operator-(ImageHasher &i) {
//    return this->hamming_distance(i);
//}
//
//int ImageHasher::operator-(const uint64_t h) {
//    return this->hamming_distance(h);
//}
