#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  std::ifstream file(model_path, std::ios::binary | std::ios::ate);

  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << model_path << std::endl;
    return 1;
  }

  std::streamsize size = file.tellg();
  file.close();

  if (size <= 0) {
    std::cerr << "Error: File is empty" << std::endl;
    return 1;
  }

  std::cout << "Model " << model_path << " verified. Size: " << size
            << " bytes." << std::endl;
  return 0;
}
