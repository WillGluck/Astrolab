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

using namespace std;

int main() {

	AstrolabPipeline pipeline("C:\\images_training_rev1\\", 50);
	pipeline.execute(ImageProcessing);

}
