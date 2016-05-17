#include "stdafx.h"

#include "AstrolabImageProcessor.h"
#include "AstrolabUtils.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>

void AstrolabImageProcessor::denoise(cv::Mat &image) {

	//Cropp.
	image = image(cv::Rect(106, 106, 212, 212));
	//cv::GaussianBlur(image, image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

	cv::Mat x = cv::Mat::ones(10, 30, CV_8U);
	cv::Mat gray;
	cv::Mat binary;
	cv::Mat final_image;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours;

	int bigger_contour_index = 0;

	//Convert the image to Gray
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::fastNlMeansDenoising(gray, gray);

	//Threshold
	threshold(gray, binary, 20, 255, CV_THRESH_BINARY);

	//Find contours
	findContours(binary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	//Find bigger contour
	for (size_t i = 0; i < contours.size(); ++i) {
		double area = cv::contourArea(contours[i]);
		if (area < 1e2 || 1e5 < area) continue;

		//cv::drawContours(bordered, contours, i, cv::Scalar(0, 0, 255), 2, 8, hierarchy, 0);
		if (cv::contourArea(contours[i]) > cv::contourArea(contours[bigger_contour_index])) {
			bigger_contour_index = i;
		}
	}

	//Create temporary image that will hold the mask
	cv::Mat mask_image(image.size(), CV_8U, cv::Scalar(0));
	//Draw your contour in mask
	drawContours(mask_image, contours, bigger_contour_index, cv::Scalar(255), CV_FILLED);

	gray.copyTo(final_image, mask_image);
	cv::blur(final_image, final_image, cv::Size(3, 3));
	final_image.convertTo(final_image, -1, 1.1, 0); //increase the contrast (double)

	//cv::Mat canny;
	//cv::Canny(finalImage, finalImage, 0, 50, 3);

	image = final_image;
}