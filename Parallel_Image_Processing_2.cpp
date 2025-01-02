#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

using namespace std;


std::vector<std::vector<float>> readKernel(const string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the kernel file." << std::endl;
        exit(1);
    }

    std::vector<std::vector<float>> kernel;
    std::string line;
    while (getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        float val;
        while (iss >> val) {
            row.push_back(val);
        }
        kernel.push_back(row);
    }

    return kernel;
}

void applyConvolution(const cv::Mat& orig_img, cv::Mat& out_img, const std::vector<std::vector<float>>& kernel) {
    int kernelRadX = kernel.size() / 2;
    int kernelRadY = kernel[0].size() / 2;

    out_img = orig_img.clone();

    for (int i = kernelRadX; i < orig_img.rows - kernelRadX; ++i) {
        for (int j = kernelRadY; j < orig_img.cols - kernelRadY; ++j) {
            float valB = 0.0, valG = 0.0, valR = 0.0;

            for (int m = -kernelRadX; m <= kernelRadX; ++m) {
                for (int n = -kernelRadY; n <= kernelRadY; ++n) {
                    cv::Vec3b color = orig_img.at<cv::Vec3b>(i + m, j + n);

                    valB += color[0] * kernel[kernelRadX + m][kernelRadY + n];
                    valG += color[1] * kernel[kernelRadX + m][kernelRadY + n];
                    valR += color[2] * kernel[kernelRadX + m][kernelRadY + n];
                }
            }

            out_img.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(valB);
            out_img.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(valG);
            out_img.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(valR);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Command line parameters must include the image file and the kernel file." << std::endl;
        return -1;
    }

    string imgfile = argv[1];
    string kernelfile = argv[2];

    cv::Mat image = cv::imread(imgfile, 1);   // Read the file
    if (!image.data) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    std::vector<std::vector<float>> kernel = readKernel(kernelfile);
    cv::Mat processed_image;
    applyConvolution(image, processed_image, kernel);

    cv::imwrite("part_1_processed_image.jpg", processed_image);  // Save the processed image

    // cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);   // Create a window for display
    // cv::imshow("Original Image", image);                      // Show our image inside it

    // cv::namedWindow("Processed Image", cv::WINDOW_AUTOSIZE);  // Create display window
    // cv::imshow("Processed Image", processed_image);           // Show our image inside it

    // cv::waitKey(10000);   // Wait 10 seconds before closing image (or a keypress to close)

    return 0;
}

