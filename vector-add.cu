#include <iostream>
#include <math.h>

__global__
void init(int n, float *x, float val){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for(int i = index; i < n; i += stride){
        x[i] = val;
    }
}

__global__
void add(int n, float *x, float *y){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i += stride){
        y[i] = x[i] + y[i];
    }
}

int main(){
    int N = 1 << 20;
    float *x;
    float *y;
    cudaMalloc(&x, sizeof(float)*N);
    cudaMalloc(&y, sizeof(float)*N);
    
    int threads_per_block = 512;
    int block_num = (N + threads_per_block - 1) / threads_per_block;
    
    init<<<block_num, threads_per_block>>>(N, x, 1.0f);
    init<<<block_num, threads_per_block>>>(N, y, 2.0f);

    add<<<block_num, threads_per_block>>>(N, x, y);
    
    float *x_host = new float[N];
    float *y_host = new float[N];

    cudaMemcpy(x_host, x, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(y_host, y, sizeof(float)*N, cudaMemcpyDeviceToHost);

    // sanity check
    float max_error = 0.0f;
    for(int i = 0; i < N; i++){
        max_error = fmax(max_error, fabs(y_host[i] - 3.0f));
    }
    
    std::cout << "Max error: " << max_error << std::endl;

    cudaFree(x);
    cudaFree(y);
    
    delete[] x_host;
    delete[] y_host;

    return 0;
}