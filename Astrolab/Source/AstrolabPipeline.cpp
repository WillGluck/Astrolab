#include "stdafx.h"

#include <string>
#include <vector>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "AstrolabPipeline.h"
#include "AstrolabNeuralNetwork.h"
#include "AstrolabFileSystem.h"

using namespace std;

AstrolabPipeline::AstrolabPipeline(std::string imagesPath, int imagesCount) :
	imagesPath(imagesPath), 
	imagesCount(imagesCount) {
}

int AstrolabPipeline::execute(PipelineAction action) {
	switch (action) {
	case ImageProcessing:
		return executeImageProcess();
	case NeuralNetwork:
		return executeNeuralNetwork();
	case FullPipeline:
		return executeFullPipeline();
	}
	return 0;
}

int AstrolabPipeline::executeImageProcess() {

	vector<string> filenames = fileSystem.getFileNamesFromPath(imagesPath);
	vector<cv::Mat> images;

	for (std::vector<int>::size_type i = 0; i < imagesCount; i++) {		
		images.push_back(cv::imread(imagesPath + filenames[i]));
		imageProcessor.process(images.back());
	}
	return 0;
}

int AstrolabPipeline::executeNeuralNetwork() {
	//TODO
	return 0;
}

int AstrolabPipeline::executeFullPipeline() {
	return executeImageProcess() == 0 ? executeNeuralNetwork() : 1;
}