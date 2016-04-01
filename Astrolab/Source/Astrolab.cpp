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
#include "AstrolabPipeline.h"
#include "AstrolabNeuralNetwork.h"
#include "AstrolabFileSystem.h"

using namespace cv;
using namespace std;

Mat src, src_gray, dest;
char* window_name = "Threshold Demo";

int main() {

	AstrolabPipeline pipeline("C:\\images_training_rev1\\", 1);
	pipeline.execute(ImageProcessing);

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
