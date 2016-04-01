#pragma once

#include <string>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "AstrolabFileSystem.h"
#include "AstrolabImageProcessor.h"
#include "AstrolabNeuralNetwork.h"

enum PipelineAction {
	ImageProcessing,
	NeuralNetwork,
	FullPipeline
};

class AstrolabPipeline {

private:

	std::string imagesPath;
	int imagesCount;

	AstrolabFileSystem fileSystem;
	
	AstrolabImageProcessor imageProcessor;
	AstrolabNeuralNetwork neuralNetwork;

	char* window_name = "Astrolab Demo";

	int executeImageProcess();

	int executeNeuralNetwork();

	int executeFullPipeline();

public:

	AstrolabPipeline(std::string initialPath, int imagesCount);
		
	int execute(PipelineAction action);

};