#include "stdafx.h"
#include "AstrolabNeuralNetwork.h"

#include "fann.h"

void AstrolabNeuralNetwork::train() {


	//Hu moments
	//cv::Moments mom = cv::moments(final_image); //Gray scale? Bordas? ....
	//double hu[7];
	//cv::HuMoments(mom, hu);

	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = (const float) 0.001;
	const unsigned int max_epochs = 500000;
	const unsigned int epochs_between_reports = 1000;

	struct fann *ann = fann_create_standard(num_layers, num_input,
		num_neurons_hidden, num_output);




}