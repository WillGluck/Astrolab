// Astrolab.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <dirent.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "AstrolabNeuralNetwork.h"
#include "AstrolabFileSystem.h"

using namespace cv;
using namespace std;

Mat src, src_gray, dest;
char* window_name = "Threshold Demo";

int main() {

	AstrolabFileSystem fs;
	string folder = "C:\\images_training_rev1\\";
	vector<string> filenames = fs.getFileNamesFromPath(folder);
	vector<Mat> images;
	for (std::vector<int>::size_type i = 0; i != 1000; i++) {
		src = imread(folder + filenames[i]);
		//Denoise
		GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
		//Convert the image to Gray
		cvtColor(src, src_gray, CV_BGR2GRAY);
		///imshow(window_name, src_gray);
		images.push_back(src);
	}
	/*
	for (auto const& filename : filenames) {
		src = imread(folder + filename);
		//Denoise
		GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
		//Convert the image to Gray
		cvtColor(src, src_gray, CV_BGR2GRAY);
		///imshow(window_name, src_gray);
	}
	*/

	/// Wait until user finishes program
	while (true)
	{
		int c;
		c = waitKey(20);
		if ((char)c == 27)
		{
			break;
		}
	}

}
