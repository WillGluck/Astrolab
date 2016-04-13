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
#include "AstrolabUtils.h"

void print(std::string text) {
	std::cout << text << std::endl;
}

void readTo(std::string &buffer) {
	std::cin >> buffer;
}

int main() {

	AstrolabPipeline pipeline;

	print("astrolab");
	std::string denoise = "denoise";
	std::string train = "train";
	std::string input;

	while (true) {
		print("select a action");
		try {
			readTo(input);
			if (input == denoise || input == train || input == "classify") {
				
				std::string path;
				std::string count;

				print("path");
				readTo(path);
				print("images count");
				readTo(count);

				if (AstrolabUtils::isInteger(count) && !path.empty()) {

					pipeline.setup(path, std::stoi(count));

					if (input == denoise) {
						pipeline.execute(Denoise);
					}
					else if (input == train) {
						pipeline.execute(Train);
					}
				}
				else {					
					print("invalid Parameters");
				}
			}
			else if (input == "exit") {
				break;
			}
			else {
				print("invalid Option");
			}
		}
		catch (std::string err) {
			print(err);
		}
		catch (...) {
			print("unexpected error");
		}
	}
}
