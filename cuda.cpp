#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

const int N = 1024;
const int BLOCK_SIZE = 1024;

// 初始化函数，用于生成随机矩阵
void init(float* A)
{
    // 创建伪随机数生成器
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

    // 设置种子
    curandSetPseudoRandomGeneratorSeed(generator, time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            A[i * N + j] = 0;
        }
        A[i * N + i] = 1.0;

        for (int j = i + 1; j < N; j++)
        {
            // 生成范围在 [0, 1) 的随机数
            curandGenerateUniform(generator, &A[i * N + j]);
        }
    }

    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] += A[k * N + j];
            }
        }
    }

    // 归一化
    for (int i = 0; i < N; i++)
    {
        float sum = 0;
        for (int j = 0; j < N; j++)
        {
            sum += A[i * N + j];
        }
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] /= sum;
        }
    }

    // 销毁生成器
    curandDestroyGenerator(generator);
}

__global__ void division_kernel(float* data, int k, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // 计算线程索引
    int element = data[k * N + k];
    int temp = data[k * N + tid];
    data[k * N + tid] = (float)temp / element;
    return;
}

__global__ void eliminate_kernel(float* data, int k)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx == 0)
        data[k * N + k] = 1.0; // 对角线元素设为 1
    int row = k + 1 + blockIdx.x; // 每个块负责一行
    while (row < N)
    {
        int tid = threadIdx.x;
        while (k + 1 + tid < N)
        {
            int col = k + 1 + tid;
            float temp_1 = data[(row * N) + col];
            float temp_2 = data[(row * N) + k];
            float temp_3 = data[k * N + col];
            data[(row * N) + col] = temp_1 - temp_2 * temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads(); // 块内同步
        if (threadIdx.x == 0)
        {
            data[row * N + k] = 0;
        }
        row += gridDim.x;
    }
}

int main()
{
    float* A = new float[N * N];
    init(A); // 初始化矩阵

    float* temp = new float[N * N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            temp[i * N + j] = A[i * N + j];
        }
    }

    float* gpudata;
    float* result = new float[N * N];
    int size = N * N * sizeof(float);
    cudaMalloc(&gpudata, size); // 分配显存空间
    cudaMemcpy(gpudata, temp, size, cudaMemcpyHostToDevice); // 将数据传输至 GPU 端

    dim3 dimBlock(BLOCK_SIZE, 1); // 线程块
    dim3 dimGrid(1, 1); // 线程网格

    cudaEvent_t start, stop; // 计时器
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); // 开始计时

    for (int k = 0; k < N; k++)
    {
        division_kernel<<<dimGrid, dimBlock>>>(gpudata, k, N); // 负责除法任务的核函数
        cudaDeviceSynchronize(); // CPU 与 GPU 之间的同步函数
        eliminate_kernel<<<dimGrid, dimBlock>>>(gpudata, k); // 负责消去任务的核函数
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // 停止计时
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU_LU: %f ms\n", elapsedTime);

    cudaMemcpy(result, gpudata, size, cudaMemcpyDeviceToHost); // 将数据传回 CPU 端

    cudaFree(gpudata); // 释放显存空间
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 清理内存
    delete[] A;
    delete[] temp;
    delete[] result;

    return 0;
}
