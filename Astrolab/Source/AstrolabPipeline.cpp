#include "stdafx.h"

#include <string>
#include <vector>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "AstrolabPipeline.h"
#include "AstrolabNeuralNetwork.h"
#include "AstrolabUtils.h"

using namespace std;

AstrolabPipeline::AstrolabPipeline() { }

void AstrolabPipeline::setup(std::string imagesPath, int imagesCount) {
	this->images_path = imagesPath;
	this->images_count = imagesCount;
}

int AstrolabPipeline::execute(PipelineAction action) {
	switch (action) {
	case Denoise:
		return denoiseImage();
	case Train:
		return 0;
	case Classify:
		return 0;
	case FullPipeline:
		return executeFullPipeline();
	}
	return 0;
}

int AstrolabPipeline::denoiseImage() {

	vector<string> filenames = AstrolabUtils::getFileNamesFromPath(images_path);
	vector<cv::Mat> images;

	for (std::vector<int>::size_type i = 0; i < images_count; i++) {		
		cv::Mat image = cv::imread(images_path + filenames[i]);
		image_processor.denoise(image);
		cv::imshow("teste", image);
		images.push_back(image);
	}

	int index{ 0 };
	std::string newPath = images_path + "..\\denoised_images\\";
	AstrolabUtils::createFolder(newPath);
	for (cv::Mat image : images) {
		cv::imwrite(newPath + filenames[index], image);
		++index;
	}

	return 0;
}

int AstrolabPipeline::trainNeuralNetwork() {
	//TODO
	return 0;
}

int AstrolabPipeline::classifyImages() {
	//TODO
	return 0;
}

int AstrolabPipeline::executeFullPipeline() {
	//TODO
	return 0;
}