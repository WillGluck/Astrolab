#pragma once

#include <vector>
#include <string>

class AstrolabUtils {

private:

public:

	static void createFolder(std::string folder_name);

	static const char* window_name;

	static std::vector<std::string> getFileNamesFromPath(std::string path);

	static void waitEnter();

	static std::string type2str(int type);

	static bool isInteger(const std::string & s);

};