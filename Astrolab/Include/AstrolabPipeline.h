#pragma once

#include <string>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "AstrolabUtils.h"
#include "AstrolabImageProcessor.h"
#include "AstrolabNeuralNetwork.h"

enum PipelineAction {
	Denoise,
	Train,
	Classify,
	FullPipeline
};

class AstrolabPipeline {

private:

	std::string images_path;
	int images_count;
	
	AstrolabImageProcessor image_processor;
	AstrolabNeuralNetwork neural_network;

	void denoiseImage();

	void trainNeuralNetwork();

	void classifyImages();

	void executeFullPipeline();

public:

	AstrolabPipeline();
		
	void execute(PipelineAction action);

	void setup(std::string initialPath, int imagesCount);

};