#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <future>
#include <utility>
#include <unordered_set>

using namespace std;
namespace fs = filesystem;
namespace po = boost::program_options;

constexpr auto NAME = "image_similarity";
constexpr auto VERSION = "v1.1.0";
constexpr auto DIM = 8;
const unordered_set<string> SUPPORTED_IMG_TYPES = {
    // Always supported
    ".bmp", ".dib", ".gif", ".pbm", ".pgm", ".ppm",
    ".pxm", ".pnm", ".sr", ".ras", ".hdr", ".pic",
    // Conditionally supported
    ".jpeg", ".jpg", ".jpe", ".png", ".webp", ".avif",
    ".jp2", ".pfm", ".tiff", ".tif", ".exr"
};

cv::Mat& normalize_image(cv::Mat &);
uint64_t average_hash(const cv::Mat &);
int hamming_distance(uint64_t);
pair<string, uint64_t> img_process_pipeline(const string);

int main(int argc, char *argv[]) {
    // Add arguments
    po::options_description desc("Usage:");
    desc.add_options()
        ("help,h", "Produce help message")
        ("version,v", "Show version")
        ("img", "Image file")
        ("comp", "Comparison file/directory")
        ("alg,a", po::value<string>()->default_value("avg"), "Hashing algorithm: avg | pcep")
    ;

    // Specify positional arguments
    po::positional_options_description p;
    p.add("img", 1);
    p.add("comp", 1);
    
    // Get arguments
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
    
    if(vm.count("help")) {
        cout << desc << endl;
        return 0;
    }
    if(vm.count("version")) {
        cout << NAME << " (ver. " << VERSION << ")" << endl;
        return 0;
    }
    if(!vm.count("img") || !vm.count("comp")) {
        cerr << "Usage: image_similarity <img1> <dir>\n";
        return 1;
    }

	// Get paths
    string img = vm["img"].as<string>();
    string comp = vm["comp"].as<string>();
	fs::path img_path(img);
	fs::path comp_path(comp);
    
    if(!fs::exists(img_path) && !fs::is_regular_file(img_path)) {
        cerr << "Enter valid img file!\n";
        return 2;
    } else if(!SUPPORTED_IMG_TYPES.count(img_path.extension().string())) {
        cerr << "Enter supported img file!\n";
        return 3;
    }
    if(!fs::is_regular_file(comp_path) && !fs::is_directory(comp_path)) {
        cerr << "Enter valid comp img file/directory!\n";
        return 2;
    }
    
    // Stores pair with string and future that returns hash
    vector<future<pair<string, uint64_t>>> futures;
    
    uint64_t ref_hash = img_process_pipeline(img).second;
    if (fs::is_regular_file(comp_path)) {
        if(!fs::exists(comp_path) || !SUPPORTED_IMG_TYPES.count(comp_path.extension().string())) {
            cerr << "Enter supported img file for comparison!\n";
            return 3;
        }
        uint64_t comp_hash = img_process_pipeline(comp).second;
        int res = hamming_distance(ref_hash ^ comp_hash);
        if (res < 8) {
            cout << img << " and " << comp << " are similar!\n";
        } else {
            // cout << img << " and " << comp << " are NOT similar!\n";
        }
    } else if(fs::is_directory(comp_path)) {
        fs::recursive_directory_iterator d(comp_path);
        fs::recursive_directory_iterator end;
        while(d != end) {
            // Skip file if does not exist or is not valid image
            if(fs::exists(d->path()) && fs::is_regular_file(d->path()) && SUPPORTED_IMG_TYPES.count(d->path().extension().string()) && d->path() != img_path) {
                string p = d->path().string();
                futures.emplace_back(async(launch::async, img_process_pipeline, p));
            }
            d++;
        }
        for(future<pair<string, uint64_t>> &fut : futures) {
            pair<string, uint64_t> path_hash = fut.get();
            int res = hamming_distance(ref_hash ^ path_hash.second);
            if (res < 8) {
                cout << img << " and " << path_hash.first << " are similar!\n";
            } else {
                // cout << path_hash.first << ": " << path_hash.second << endl;
                // cout << img << " and " << path_hash.first << " are NOT similar!\n";
            }
        }
    }
	return 0;
}

cv::Mat& normalize_image(cv::Mat &img) {
    // img is assigned a new object in new memory address
    if (img.type() == CV_32FC1) {
        cv::Mat out1, out2;
        cv::normalize(img, out1, 0.0, 1.0, cv::NORM_MINMAX);
        out1.convertTo(out2, CV_8U, 255.0);
        img = out2.clone();
    }
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

pair<string, uint64_t> img_process_pipeline(const string img) {
    // Load image
    cv::Mat imgcv = cv::imread(img);
    
    // Resize
    cv::Mat resized_img;
    cv::resize(imgcv, resized_img, cv::Size(DIM, DIM));
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(resized_img, gray, cv::COLOR_BGR2GRAY);
    
    // Normalize if necessary
    normalize_image(gray);
    
    // Return hash
    return pair(img, average_hash(gray));
}
