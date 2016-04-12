#include "stdafx.h"

#include "AstrolabImageProcessor.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"

#include "Astrolab.h"
#include <iostream>


int AstrolabImageProcessor::process(cv::Mat& image) {
	
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = (const float) 0.001;
	const unsigned int max_epochs = 500000;
	const unsigned int epochs_between_reports = 1000;

	image = image(cv::Rect(106, 106, 212, 212));
	cv::GaussianBlur(image, image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

	cv::Mat gray;
	cv::Mat binary;
	cv::Mat finalImage;
	vector<cv::Vec4i> hierarchy;
	vector<vector<cv::Point> > contours;

	int biggerContourIndex = 0;

	/*cv::imshow(window_name, image);
	astro::waitEnter();*/

	//Denoise/Edge detections/Spike Removal.
	//Convert the image to Gray

	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::fastNlMeansDenoising(gray, gray);


	//Gauss
	/*cv::imshow(window_name, image);*/

	threshold(gray, binary, 20, 255, CV_THRESH_BINARY);
	//cv::imshow(window_name, binary);

	/// Find contours
	//cout << astro::type2str(image.type());

	findContours(binary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	//TEMP
	for (size_t i = 0; i < contours.size(); ++i) {
		double area = cv::contourArea(contours[i]);
		if (area < 1e2 || 1e5 < area) continue;

		//cv::drawContours(bordered, contours, i, cv::Scalar(0, 0, 255), 2, 8, hierarchy, 0);
		if (cv::contourArea(contours[i]) > cv::contourArea(contours[biggerContourIndex])) {
			biggerContourIndex = i;
		}
	}

	// create temporary image that will hold the mask
	cv::Mat mask_image(image.size(), CV_8U, cv::Scalar(0));
	// draw your contour in mask
	drawContours(mask_image, contours, biggerContourIndex, cv::Scalar(255), CV_FILLED);

	gray.copyTo(finalImage, mask_image);

	cv::blur(finalImage, finalImage, cv::Size(3, 3));
	imshow(window_name, finalImage);
	astro::waitEnter();

	cv::Mat canny;
	cv::Canny(finalImage, canny, 0, 50, 3);

	//Hu moments
	cv::Moments mom = cv::moments(finalImage); //Gray scale? Bordas? ....
	double hu[7];
	cv::HuMoments(mom, hu);
	
	astro::waitEnter();

	return 0;
}