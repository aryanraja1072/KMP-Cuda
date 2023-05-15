#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        printf("Device %d supports mapping host memory:  ", device);
        if (deviceProp.canMapHostMemory)
            printf("Yes\n");
        else
            printf("No\n");
    }

    return 0;
}

