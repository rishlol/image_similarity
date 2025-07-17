#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <thread>
#include <vector>
#include <cstdint>

using namespace std;
namespace fs = filesystem;

constexpr auto DIM = 8;

cv::Mat normalize_image(cv::Mat img) {
    cv::Mat out;
    cv::normalize(img, out, 0.0, 1.0, cv::NORM_MINMAX);
    out.convertTo(img, CV_8U, 255.0);
    return img;
}

uint64_t average_hash(const cv::Mat &img) {
    // Expects img to be of type CV_8UC1
    double avg = cv::mean(img)[0];
    
    // Build 64-bit hash
    uint64_t hash = 0;
    for (int r = 0; r < DIM; r += 1) {
        for (int c = 0; c < DIM; c += 1) {
            hash <<= 1;
            if (img.at<uchar>(r, c) >= avg) {
                hash |= 1;
            }
        }
    }
    return hash;
}

int hamming_distance(uint64_t diff) {
    int count = 0;
    while(diff > 0) {
        count += diff & 1;
        diff >>= 1;
    }
    return count;
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		cerr << "Usage: image_similarity <img1> <img2>";
		return 1;
	}

	// Get image paths
	fs::path path1(argv[1]);
	fs::path path2(argv[2]);
	string strpath1 = path1.string();
	string strpath2 = path2.string();

	// Load images
	cv::Mat img1 = cv::imread(strpath1);
	cv::Mat img2 = cv::imread(strpath2);

	// Resize to 8x8
	cv::Mat img1_8, img2_8;
	cv::resize(img1, img1_8, cv::Size(DIM, DIM));
	cv::resize(img2, img2_8, cv::Size(DIM, DIM));

	// Convert to grayscale
	cv::Mat gray1, gray2;
	cv::cvtColor(img1_8, gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img2_8, gray2, cv::COLOR_BGR2GRAY);
    
    // Normalize images if necessary
    if (gray1.type() == CV_8UC1) {
        gray1 = normalize_image(gray1);
    }
    if (gray2.type() == CV_8UC1) {
        gray2 = normalize_image(gray2);
    }
    
    // Get hashes
    uint64_t hash1 = average_hash(gray1), hash2 = average_hash(gray2);
    uint64_t comp = hash1 ^ hash2;

	// Compare images
    int ne = hamming_distance(comp);
    bool similar = ne == 0;
    
//    cout << "Difference " << ne << endl;
	if (similar) {
		cout << strpath1 << " and " << strpath2 << " are similar!\n";
	} else {
		cout << strpath1 << " and " << strpath2 << " are NOT similar\n";
	}

//	cv::imshow("img 1", gray1);
//	cv::imshow("img 2", gray2);
//	cv::waitKey(0);

	return 0;
}
