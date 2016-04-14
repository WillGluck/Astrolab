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

void AstrolabPipeline::execute(PipelineAction action) {
	switch (action) {
	case Denoise:
		denoiseImage(); break;
	case Train:
		trainNeuralNetwork(); break;
	case Classify:
		classifyImages(); break;
	case FullPipeline:
		executeFullPipeline(); break;
	}
}

void AstrolabPipeline::denoiseImage() {

	std::string newPath = images_path + "..\\denoised_images\\";
	AstrolabUtils::createFolder(newPath);
	vector<string> filenames = AstrolabUtils::getFileNamesFromPath(images_path);

	for (std::vector<int>::size_type i = 0; i < images_count; i++) {		
		cv::Mat image = cv::imread(images_path + filenames[i]);
		image_processor.denoise(image);
		cv::imwrite(newPath + "d_" + filenames[i], image);
	}
}

void AstrolabPipeline::trainNeuralNetwork() {
	//TODO
}

void AstrolabPipeline::classifyImages() {
	//TODO
}

void AstrolabPipeline::executeFullPipeline() {
	//TODO
}