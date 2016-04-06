#pragma once

#include "opencv2/imgproc/imgproc.hpp"

#include "AstrolabFileSystem.h"

class AstrolabImageProcessor {

private:
	char* window_name = "Astrolab Demo";
	AstrolabFileSystem fileSystem;

public:

	int process(cv::Mat& image);
		
};