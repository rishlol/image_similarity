#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <thread>
#include <vector>
#include <mutex>

using namespace std;
namespace fs = filesystem;

constexpr auto DIM = 8;
mutex sum_mutex;

void add_pixels(const cv::Mat m, double& sum) {
	double local_sum = 0;
	switch(m.type()) {
	case CV_8UC1:
		for (int i = 0; i < m.cols; i += 1) {
			local_sum += m.at<uchar>(0, i);
		}
		break;
	case CV_32FC1:
		for (int i = 0; i < m.cols; i += 1) {
			local_sum += m.at<float>(0, i);
		}
		break;
	default:
		cerr << "Unsupported Mat type in add_pixels\n";
		return;
	}

	lock_guard<mutex> lock(sum_mutex);
	sum += local_sum;
}

void threshold(cv::Mat& m, const double avg) {
	for (int r = 0; r < m.rows; r += 1) {
		for (int c = 0; c < m.cols; c += 1) {
			switch (m.type()) {
			case CV_8UC1:
				m.at<uchar>(r, c) = m.at<uchar>(r, c) >= avg ? 255 : 0;
				break;
			case CV_32FC1:
				m.at<float>(r, c) = m.at<float>(r, c) >= avg ? 1.0 : 0;
				break;
			default:
				cerr << "Unsupported Mat type in threshold\n";
				return;
			}
		}
	}
}

cv::Mat normalize_image(cv::Mat img) {
	cv::Mat out;
	cv::normalize(img, out, 0.0, 1.0, cv::NORM_MINMAX);
	out.convertTo(img, CV_8U, 255.0);
	return img;
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

	// Get averages
	double sum1 = 0, sum2 = 0;
	vector<thread> threads;
	for (int r = 0; r < DIM; r += 1) {
		threads.emplace_back(thread(add_pixels, gray1.row(r), std::ref(sum1)));
		threads.emplace_back(thread(add_pixels, gray2.row(r), std::ref(sum2)));
	}

	for (auto& thread : threads) {
		thread.join();
	}
	double avg1 = sum1 / gray1.total(), avg2 = sum2 / gray2.total();

	// Normalize images if necessary
	if (gray1.type() == CV_8UC1) {
		gray1 = normalize_image(gray1);
	}
	if (gray2.type() == CV_8UC1) {
		gray2 = normalize_image(gray2);
	}

	// Threshold
	thread t1 = thread(threshold, std::ref(gray1), avg1);
	thread t2 = thread(threshold, std::ref(gray2), avg2);
	t1.join();
	t2.join();

	// Compare images
	cv::Mat res(DIM, DIM, CV_8UC1);
	cv::compare(gray1, gray2, res, cv::CMP_NE);
	int ne = cv::countNonZero(res);
	bool similar = ne == 0;

	if (similar) {
		cout << strpath1 << " and " << strpath2 << " are similar!\n";
	} else {
		cout << strpath1 << " and " << strpath2 << " are NOT similar\n";
	}

	//cv::Mat img1_test;
	//cv::resize(img1, img1_test, cv::Size(img1.cols / 2, img1.rows / 2));
	//cout << gray1.rows << " " << gray1.cols << endl;
	//cout << avg1 << " " << avg2 << endl;
	//cout << similar << endl;
	//cv::imshow("img 1", gray1);
	//cv::imshow("img 2", gray2);
	//cv::waitKey(0);

	return 0;
}
