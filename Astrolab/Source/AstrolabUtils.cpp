#include "stdafx.h"
#include "AstrolabUtils.h"

#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <fstream>
#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>

const char* window_name{ "Astrolab Demo" };

void AstrolabUtils::createFolder(std::string folder_name) {
	system(("mkdir " + folder_name).c_str());
}

void AstrolabUtils::waitEnter() {
	while (true)
	{
		int c;
		c = cv::waitKey(1);
		if ((char)c == 27)
		{
			break;
		}
	}
}

std::string AstrolabUtils::type2str(int type)
{
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

bool AstrolabUtils::isInteger(const std::string & s)
{
	if (s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false;

	char * p;
	strtol(s.c_str(), &p, 10);

	return (*p == 0);
}

std::vector<std::string> AstrolabUtils::getFileNamesFromPath(std::string path) {

	DIR    *dir;
	dirent *pdir;
	std::vector<std::string> files;

	dir = opendir(path.c_str());
	while (pdir = readdir(dir))
	{
		if (strlen(pdir->d_name) > 3) {
			files.push_back(pdir->d_name);
		}
	}
	return files;
}
