#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <mpi.h>

using namespace std;
using namespace cv;

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

// Taken from lab 2 manual
void parallelRange(int globalstart, int globalstop, int irank, int nproc, int &localstart, int &localstop, int &localcount) { //parallelRange
    int nrows = globalstop - globalstart + 1;
    int divisor = nrows / nproc;
    int remainder = nrows % nproc;
    int offset;

    if (irank < remainder) offset = irank;
    else offset = remainder;

    localstart = irank * divisor + globalstart + offset;
    localstop = localstart + divisor - 1;

    if (remainder > irank) localstop += 1;
    localcount = localstop - localstart + 1;
} //parallelRange

int main(int argc, char** argv) { //main
    MPI_Init(&argc, &argv);
    int rank;
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cv::Mat image;
    std::vector<std::vector<float>> kernel;

    if (argc < 3) {
        std::cerr << "Command line parameters must include the image filename and the kernel file." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    string filename = argv[1];
    string kernelfile = argv[2];

    int M, N;
    int m;
    if (rank == 0) {
        image = cv::imread(filename, 1);
        if (!image.data) {
            std::cerr << "Could not open or find the image" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        M = image.rows;
        N = image.cols;
        kernel = readKernel(kernelfile);
        m = kernel.size() / 2; // Assuming kernel is square
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        image = cv::Mat::zeros(M, N, CV_8UC3);
        //kernel.resize(kernel.size(), std::vector<float>(kernel[0].size()));
        kernel.resize(2 * m + 1, std::vector<float>(2 * m + 1));
    }

    // Flatten kernel for broadcasting
    std::vector<float> kernel_flat((2 * m + 1) * (2 * m + 1));
    if (rank == 0) {
        for (int i = 0; i < kernel.size(); ++i)
            std::copy(kernel[i].begin(), kernel[i].end(), kernel_flat.begin() + i * kernel.size());
    }
    MPI_Bcast(kernel_flat.data(), kernel_flat.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Reconstruct kernel in non-root processes
    if (rank != 0) {
        for (int i = 0; i < 2 * m + 1; ++i)
            for (int j = 0; j < 2 * m + 1; ++j)
                kernel[i][j] = kernel_flat[i * (2 * m + 1) + j];
    }

    // Calculate ranges
    int localstart, localstop, localcount;
    parallelRange(0, M - 1, rank, nproc, localstart, localstop, localcount);

    // Offset localstart and localstop for halo
    if (rank == 0) {
        localstop += m; // first rank has halo only at the bottom
    } else if (rank == nproc - 1) {
        localstart -= m; // last rank has halo only at the top
    } else {
        localstart -= m;
        localstop += m; // other ranks have both top and bottom halos
    }

    // Sub-images including halos
    int localrows = localstop - localstart + 1;
    cv::Mat subImage(localrows, N, CV_8UC3);

    // string filename0 = "unblur_subImage_" + std::to_string(rank) + ".png";
    // cv::imwrite(filename0,subImage);

    std::vector<int> sendcounts(nproc), displs(nproc);
    for (int i = 0; i < nproc; ++i) {
        int start, stop, count;
        parallelRange(0, M - 1, i, nproc, start, stop, count);
        sendcounts[i] = (count + ((i == 0 || i == nproc - 1) ? m : 2 * m)) * N * 3;
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + sendcounts[i - 1]);
    }

    MPI_Scatterv((rank == 0) ? image.ptr() : nullptr, sendcounts.data(), displs.data(), MPI_BYTE, subImage.ptr(), sendcounts[rank], MPI_BYTE, 0, MPI_COMM_WORLD);
    //MPI_Scatterv(image.ptr(), sendcounts.data(), displs.data(), MPI_BYTE, subImage.ptr(), sendcounts[rank], MPI_BYTE, 0, MPI_COMM_WORLD);

    cv::Mat processedSubImage;
    applyConvolution(subImage, processedSubImage, kernel);
    string sub_img_file = "part_4_subImage_" + std::to_string(rank) + ".jpg";
    cv::imwrite(sub_img_file,processedSubImage);

    int nonHaloRows = localcount;
    if (rank == 0) {
        nonHaloRows -= m; // remove bottom halo
    } else if (rank == nproc - 1) {
        nonHaloRows -= m; // remove top halo
    } else {
        nonHaloRows -= 2 * m; // remove both top and bottom halos
    }

    for (int iRank = 0; iRank < nproc; ++iRank) {
        int start, stop, count;
        parallelRange(0, M - 1, iRank, nproc, start, stop, count);
        sendcounts[iRank] = count * N * 3;

        //calculateing sendcounts without halo now
        if (iRank == 0) {
            sendcounts[iRank] = (count - m) * N * 3; 
        } else if (iRank == nproc-1) {
            sendcounts[iRank] = (count - m) * N * 3; 
        } else {
            sendcounts[iRank] = (count - 2 * m) * N * 3; 
        }


        displs[iRank] = (iRank == 0) ? 0 : (displs[iRank - 1] + sendcounts[iRank - 1]);
    }

    cv::Mat gatheredImage(M, N, CV_8UC3);
    MPI_Gatherv(processedSubImage.ptr(), nonHaloRows * N * 3, MPI_BYTE, (rank == 0) ? gatheredImage.ptr() : nullptr, sendcounts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    //MPI_Gatherv(processedSubImage.ptr(), nonHaloRows * N * 3, MPI_BYTE, (rank == 0) ? gatheredImage.ptr() : nullptr, sendcounts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Display images
        cv::imshow("Original Image", image);
        cv::imshow("Processed Image", gatheredImage);
        cv::waitKey(10000);
        cv::imwrite("part_4_final_processed_image.jpg", gatheredImage);
    }

    MPI_Finalize();
    return 0;
} //main

