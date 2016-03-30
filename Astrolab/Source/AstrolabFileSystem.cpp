#include "stdafx.h"
#include "AstrolabFileSystem.h"

#include <stdlib.h>
#include <fstream>
#include <stdio.h>
#include <dirent.h>

using namespace std;

vector<string> AstrolabFileSystem::getFileNamesFromPath(string path) {
	DIR    *dir;
	dirent *pdir;
	vector<string> files;
	dir = opendir(path.c_str());
	while (pdir = readdir(dir))
	{
		if (strlen(pdir->d_name) > 3) {
			files.push_back(pdir->d_name);
		}
	}
	return files;
}