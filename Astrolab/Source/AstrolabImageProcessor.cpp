#include "stdafx.h"

#include "AstrolabImageProcessor.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "Astrolab.h"
#include <iostream>

using namespace cv;
using namespace std;

int AstrolabImageProcessor::process(cv::Mat& image) {

	cv::imshow(window_name, image);
	astro::waitEnter();

	//Convert the image to Gray
	cv::cvtColor(image, image, CV_BGR2GRAY);
	cv::imshow(window_name, image);
	astro::waitEnter();

	cv::GaussianBlur(image, image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	cv::imshow(window_name, image);
	astro::waitEnter();

	cv::threshold(image, image, 40, 255, CV_THRESH_BINARY);
	cv::imshow(window_name, image);
	astro::waitEnter();

	/// Find contours
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;

	/// Detect edges using canny
	cout << astro::type2str(image.type());
	cv::findContours(image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	astro::waitEnter();

	return 0;
}