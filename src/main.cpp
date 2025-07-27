#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <future>
#include <utility>
#include <unordered_set>
#include "ImageHasher.h"

using namespace std;
namespace fs = filesystem;
namespace po = boost::program_options;

constexpr auto NAME = "image_similarity";
constexpr auto VERSION = "v1.1.0";
constexpr int AVG_THRES = 14; // Threshold for average hashing
constexpr int PCP_THRES = 18; // Threshold for perceptual hashing
const unordered_set<string> SUPPORTED_IMG_TYPES = {
    // Always supported
    ".bmp", ".dib", ".gif", ".pbm", ".pgm", ".ppm",
    ".pxm", ".pnm", ".sr", ".ras", ".hdr", ".pic",
    // Conditionally supported
    ".jpeg", ".jpg", ".jpe", ".png", ".webp", ".avif",
    ".jp2", ".pfm", ".tiff", ".tif", ".exr"
};

namespace img_sim {
    enum HASH_TYPE {
        ERR = -1,
        AVERAGE,
        PERCEPTUAL,
    };

    inline HASH_TYPE get_hash_type(const string &s) {
        if(s == "avg")
            return AVERAGE;
        else if(s == "pcp")
            return PERCEPTUAL;
        return ERR;
    }
}

PipelineOut average_hash_pipeline_wrapper(const fs::path);
PipelineOut perceptual_hash_pipeline_wrapper(const fs::path);

int main(int argc, char *argv[]) {
    // Add arguments
    po::options_description desc("Usage");
    desc.add_options()
        ("help,h", "Produce help message")
        ("version,v", "Show version")
        ("alg,a", po::value<string>()->default_value("avg"), "Hashing algorithm: avg | pcp")
        ("img", "Image file")
        ("comp", "Comparison file/directory")
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
        cerr << "Usage: image_similarity <img> <comp>\n";
        return 1;
    }

	// Get paths
    string img = vm["img"].as<string>();
    string comp = vm["comp"].as<string>();
	fs::path img_path(img);
	fs::path comp_path(comp);
    
    // Checking img and comp inputs
    if(!fs::exists(img_path)) {
        cerr << "Image does not exist! Enter valid img file.\n";
        return 2;
    } else if(!fs::is_regular_file(img_path)) {
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
    
    img_sim::HASH_TYPE hash_func = img_sim::get_hash_type(vm["alg"].as<string>());
    ImageHasher img_data(img_path);
    switch (hash_func) {
        case img_sim::AVERAGE:
            img_data.average_hash_pipeline();
            break;
        case img_sim::PERCEPTUAL:
            img_data.perceptual_hash_pipeline();
            break;
        case img_sim::ERR:
            cerr << "Enter a valid hash function!\n";
            return 4;
    }
    
    // Stores pair with string and future that returns hash value
    vector<future<PipelineOut>> futures;

    if (fs::is_regular_file(comp_path)) {
        if(!fs::exists(comp_path) || !SUPPORTED_IMG_TYPES.count(comp_path.extension().string())) {
            cerr << "Enter supported img file for comparison!\n";
            return 3;
        }
        ImageHasher comp_data(comp_path);
        switch (hash_func) {
            case img_sim::AVERAGE:
                comp_data.average_hash_pipeline();
                break;
            case img_sim::PERCEPTUAL:
                comp_data.perceptual_hash_pipeline();
                break;
            case img_sim::ERR:
                cerr << "Enter a valid hash function!\n";
                return 4;
        }
        int res = img_data - comp_data;
        if (res < AVG_THRES) {
            cout << img << " and " << comp<< " are similar!\n";
        }
    } else if(fs::is_directory(comp_path)) {
        fs::recursive_directory_iterator d(comp_path);
        fs::recursive_directory_iterator end;
        while(d != end) {
            // Skip file if does not exist or is not valid image
            if(fs::exists(d->path()) && fs::is_regular_file(d->path()) && SUPPORTED_IMG_TYPES.count(d->path().extension().string()) && d->path() != img_path) {
                switch (hash_func) {
                    case img_sim::AVERAGE:
                        futures.emplace_back(async(launch::async, average_hash_pipeline_wrapper, d->path()));
                        break;
                    case img_sim::PERCEPTUAL:
                        futures.emplace_back(async(launch::async, perceptual_hash_pipeline_wrapper, d->path()));
                        break;
                    case img_sim::ERR:
                        cerr << "Enter a valid hash function!\n";
                        return 4;
                }
            }
            d++;
        }
        for(future<PipelineOut> &fut : futures) {
            PipelineOut path_hash = fut.get();
            int res = img_data - path_hash.second;
            switch (hash_func) {
                case img_sim::AVERAGE:
                    if (res < AVG_THRES)
                        cout << img << " and " << path_hash.first << " are similar!\n";
                    break;
                case img_sim::PERCEPTUAL:
                    if (res < PCP_THRES)
                        cout << img << " and " << path_hash.first << " are similar!\n";
                    break;
                case img_sim::ERR:
                    cerr << "Enter a valid hash function!\n";
                    return 4;
            }
        }
    }
	return 0;
}

PipelineOut average_hash_pipeline_wrapper(const fs::path img) {
    ImageHasher data(img);
    return data.average_hash_pipeline();
}

PipelineOut perceptual_hash_pipeline_wrapper(const fs::path img) {
    ImageHasher data(img);
    return data.perceptual_hash_pipeline();
}
