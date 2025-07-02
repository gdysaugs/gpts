#include <chrono>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

#include "face_restoration.hpp"

#define DEVICE 0  // GPU id

int main(int argc, char **argv) {
    cudaSetDevice(DEVICE);

    if (argc != 6 || std::string(argv[2]) != "-i" || std::string(argv[4]) != "-o") {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "For Batch Processing:" << std::endl;
        std::cerr << "./face_restoration_batch model.engine -i input_dir -o output_dir" << std::endl;
        return -1;
    }

    const std::string engine_file_path = argv[1];
    const std::string input_dir = argv[3];
    const std::string output_dir = argv[5];

    // Create output directory
    std::filesystem::create_directories(output_dir);

    FaceRestoration sample = FaceRestoration(engine_file_path);

    // Process all images in input directory
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".png" || 
            entry.path().extension() == ".jpg" || 
            entry.path().extension() == ".jpeg") {
            
            std::string input_path = entry.path().string();
            std::string filename = entry.path().filename().string();
            std::string output_path = output_dir + "/" + filename;
            
            cv::Mat img = cv::imread(input_path);
            if (img.empty()) {
                std::cerr << "Failed to read: " << input_path << std::endl;
                continue;
            }

            cv::Mat res;
            auto start = std::chrono::system_clock::now();
            sample.infer(img, res);
            auto end = std::chrono::system_clock::now();
            
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            std::cout << "Processed " << filename << " in " << elapsed / 1000.0 << " ms" << std::endl;
            
            cv::imwrite(output_path, res);
        }
    }

    std::cout << "Batch processing completed!" << std::endl;
    return 0;
}