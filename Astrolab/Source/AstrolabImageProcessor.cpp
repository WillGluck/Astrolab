#include "stdafx.h"

#include "AstrolabImageProcessor.h"
#include "opencv2/highgui/highgui.hpp"

int AstrolabImageProcessor::process(cv::Mat& image) {

	cv::GaussianBlur(image, image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	//Convert the image to Gray
	cv::cvtColor(image, image, CV_BGR2GRAY);

	return 0;
}