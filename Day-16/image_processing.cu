#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

#define THREADS_PER_BLOCK 256
#define PI 3.14159f
#define CHUNK_SIZE 256

using namespace cv;
using namespace std;

__constant__ float kx_c[CHUNK_SIZE], ky_c[CHUNK_SIZE], kz_c[CHUNK_SIZE];

__global__ void FULLHDIM(float* rphi, float* iphi, float* phiMag,
                         float* x, float* y, float* z, float* rMu, float* iMu, int M) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= M) return;

    float xnr = x[n];
    float ynr = y[n];
    float znr = z[n];

    float real_FHD_r = rphi[n];
    float img_FHD_r = iphi[n];

    for (int m = 0; m < M; m++) {
        float expFHD = 2.0f * PI * (kx_c[m] * xnr + ky_c[m] * ynr + kz_c[m] * znr);
        float carg = __cosf(expFHD);
        float sarg = __sinf(expFHD);

        real_FHD_r += rMu[m] * carg - iMu[m] * sarg;
        img_FHD_r += iMu[m] * carg + rMu[m] * sarg;
    }

    rphi[n] = real_FHD_r;
    iphi[n] = img_FHD_r;
}

int main() {
    Mat image = imread("image.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Can't find the image!" << endl;
        return -1;
    }

    image.convertTo(image, CV_32F, 1.0 / 255.0);

    int N = image.rows * image.cols;
    int M = CHUNK_SIZE;

    float *x, *y, *z, *real_Mu, *img_Mu, *real_phi, *img_phi, *phi_Mag;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&z, N * sizeof(float));
    cudaMallocManaged(&real_Mu, M * sizeof(float));  
    cudaMallocManaged(&img_Mu, M * sizeof(float));
    cudaMallocManaged(&real_phi, N * sizeof(float));
    cudaMallocManaged(&img_phi, N * sizeof(float));
    cudaMallocManaged(&phi_Mag, N * sizeof(float));

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int idx = i * image.cols + j;
            x[idx] = (float)j / image.cols;
            y[idx] = (float)i / image.rows;
            z[idx] = image.at<float>(i, j);
            real_phi[idx] = z[idx];
            img_phi[idx] = 0.0f;
        }
    }

    for (int i = 0; i < M; i++) {
        real_Mu[i] = static_cast<float>(rand()) / RAND_MAX;
        img_Mu[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaMemcpyToSymbol(kx_c, x, CHUNK_SIZE * sizeof(float));
    cudaMemcpyToSymbol(ky_c, y, CHUNK_SIZE * sizeof(float));
    cudaMemcpyToSymbol(kz_c, z, CHUNK_SIZE * sizeof(float));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    FULLHDIM<<<blocks, THREADS_PER_BLOCK>>>(real_phi, img_phi, phi_Mag, x, y, z, real_Mu, img_Mu, M);
    cudaDeviceSynchronize();

    Mat outputImage(image.rows, image.cols, CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int idx = i * image.cols + j;
            outputImage.at<float>(i, j) = sqrt(real_phi[idx] * real_phi[idx] + img_phi[idx] * img_phi[idx]);
        }
    }

    normalize(outputImage, outputImage, 0, 255, NORM_MINMAX);
    outputImage.convertTo(outputImage, CV_8U);
    imwrite("output.jpg", outputImage);

    cudaFree(x); cudaFree(y); cudaFree(z);
    cudaFree(real_Mu); cudaFree(img_Mu);
    cudaFree(real_phi); cudaFree(img_phi); cudaFree(phi_Mag);

    return 0;
}
