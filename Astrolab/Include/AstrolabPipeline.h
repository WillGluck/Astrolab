#pragma once

enum PipelineAction {
	Image,
	NeuralNetwork,
	Full
};

class AstrolabPipeline {

private:

public:
	int execute(PipelineAction action);



};