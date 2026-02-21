#include <book_interface.h>
#include <cpu_bitmap.h>
#include <iostream>

namespace ch5{
    template<int threadsPerBlock, int N>
    __global__ void gpu_dot(float *a, float *b, float *c) {
        __shared__ float cache[threadsPerBlock];
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int cacheIndex = threadIdx.x;
        float temp = 0;

        while (tid < N) {
            temp += a[tid] * b[tid];
            tid += blockDim.x * gridDim.x;
        }

        cache[cacheIndex] = temp;

        __syncthreads();

        // Could exploit locality of memory reference to speed up
        int i = blockDim.x / 2;
        while(i != 0) {
            if(cacheIndex < i) {
                cache[cacheIndex] += cache[cacheIndex + i];
            }
            __syncthreads();
            i >>= 1;
        }

        if (cacheIndex == 0)
            c[blockIdx.x] = cache[cacheIndex];
    }

    int vec_dot() {
        constexpr int N = 12 * 1024;
        constexpr int threadsPerBlock = 256;
        constexpr int blocksPerGrid = std::min(32, (N + threadsPerBlock - 1) / threadsPerBlock);
        int stat = 0;

        float *a = nullptr, *b = nullptr, sum = 0, *partial_sum = nullptr;
        float *dev_a = nullptr, *dev_b = nullptr, *dev_partial_sum = nullptr;

        auto cleanUp = [&stat, &a, &b, &partial_sum, &dev_a, &dev_b, &dev_partial_sum](void)->int{
            // Safe to free nullptr.
            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_partial_sum);

            delete[] a;
            delete[] b;
            delete[] partial_sum;

            return stat;
        };

        // CPU mem alloc
        a = new float[N];
        b = new float[N];
        partial_sum = new float[blocksPerGrid];
        // GPU mem alloc
        stat = RETURN_ERROR( cudaMalloc( (void**)&dev_a, N*sizeof(float) ) );
        if(stat) return cleanUp();
        stat = RETURN_ERROR( cudaMalloc( (void**)&dev_b, N*sizeof(float) ) );
        if(stat) return cleanUp();
        stat = RETURN_ERROR( cudaMalloc( (void**)&dev_partial_sum, blocksPerGrid*sizeof(float) ) );
        if(stat) return cleanUp();

        for (int i = 0; i < N; ++ i) {
            a[i] = i;
            b[i] = i * 2;
        }

        stat = RETURN_ERROR( cudaMemcpy( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice ) );
        if(stat) return cleanUp();
        stat = RETURN_ERROR( cudaMemcpy( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice ) );
        if(stat) return cleanUp();

        gpu_dot<threadsPerBlock, N><<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_sum);

        stat = RETURN_ERROR( cudaMemcpy( partial_sum, dev_partial_sum, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost ) );
        if(stat) return cleanUp();

        for(int i = 0; i < blocksPerGrid; ++i) {
            sum += partial_sum[i];
        }

        
        std::cout << "GPU vec dot product is: " << sum << std::endl;

        return cleanUp();
    }

    template<int DIM, int BLK>
    __global__ void gpu_blur(unsigned char* dev_bitmap, const float PI) {
        __shared__ float shared[BLK][BLK];
        constexpr float period = 128.f;
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = x + y * blockDim.x * gridDim.x;

        shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x * 2.f * PI / period) + 1.f) *
            (sinf(y * 2.f * PI / period) + 1.f) / 4.f;

        __syncthreads();

        dev_bitmap[idx * 4 + 0] = 0;
        dev_bitmap[idx * 4 + 1] = shared[BLK - 1 - threadIdx.x][BLK - 1 - threadIdx.y];
        dev_bitmap[idx * 4 + 2] = 0;
        dev_bitmap[idx * 4 + 3] = 255;
    }

    int blur_img(void) {
        constexpr int DIM = 1024;
        constexpr int BLK = 16;
        constexpr float PI = 3.1415926535897932f;
        int stat = 0;
        CPUBitmap bitmap { DIM, DIM };
        unsigned char* dev_bitmap = nullptr;
        auto cleanUp = [&stat, &dev_bitmap](void)->int{
            cudaFree(dev_bitmap);
            return stat;
        };

        stat = RETURN_ERROR( cudaMalloc((void**)&dev_bitmap, bitmap.image_size()) );
        if(stat) return cleanUp();

        dim3 grid { DIM / BLK, DIM / BLK };
        dim3 block { BLK, BLK };
        gpu_blur<DIM, BLK><<<grid, block>>>(dev_bitmap, PI);

        stat = RETURN_ERROR( cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
        if(stat) return cleanUp();

        bitmap.display_and_exit();

        return cleanUp();
    }

    int ch_5_2() {
        int stat = 0;

        stat = vec_dot();
        if(stat) return -1;

        stat = blur_img();
        if(stat) return -1;

        return stat;
    }
}