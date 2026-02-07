#include <chrono>
#include <algorithm>
#include <iostream>
#include <vector>
#include <book_interface.h>

namespace ch5 {
    template<typename T, size_t S>
    __global__ void gpu_add_kernel(T *a, T *b, T *c) {
        size_t tid = blockIdx.x;
        if (tid < S)
            c[tid] = a[tid] + b[tid];
    }

    template<typename T, size_t S>
    __global__ void gpu_add_kernel_v1(T *a, T *b, T *c) {
        size_t tid = threadIdx.x;
        if (tid < S)
            c[tid] = a[tid] + b[tid];
    }

    template<typename T, size_t S>
    __global__ void gpu_add_kernel_v2(T *a, T *b, T *c) {
        // size_t could cause potentially overflow
        long long tid = threadIdx.x + blockIdx.x * blockDim.x;
        while (tid < S) {
            c[tid] = a[tid] + b[tid];
            tid += blockDim.x * gridDim.x;
        }
    }

    template<typename T, size_t S>
    int gpu_add(T a[S], T b[S], T c[S]) {
        std::cout << "GPU vec add for pointer-like array" << std::endl;
        int stat = 0;
        T *d_a, *d_b, *d_c;

        stat = RETURN_ERROR(cudaMalloc((void**)&d_a, S * sizeof(T)));
        if(stat) return -1;
        stat = RETURN_ERROR(cudaMalloc((void**)&d_b, S * sizeof(T)));
        if(stat) return -1;
        stat = RETURN_ERROR(cudaMalloc((void**)&d_c, S * sizeof(T)));
        if(stat) return -1;

        stat = RETURN_ERROR(cudaMemcpy(d_a, a, S, cudaMemcpyHostToDevice));
        if(stat) return -1;
        stat = RETURN_ERROR(cudaMemcpy(d_b, b, S, cudaMemcpyHostToDevice));
        if(stat) return -1;

        gpu_add_kernel<T, S><<<S, 1>>>(d_a, d_b, d_c);

        stat = RETURN_ERROR(cudaMemcpy(c, d_c, S, cudaMemcpyDeviceToHost));
        if(stat) return -1;

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        return stat;
    }

    template<typename T, size_t S>
    int gpu_add_v1(T a[S], T b[S], T c[S]) {
        std::cout << "GPU vec add v1 for pointer-like array" << std::endl;
        int stat = 0;
        T *d_a, *d_b, *d_c;

        stat = RETURN_ERROR(cudaMalloc((void**)&d_a, S * sizeof(T)));
        if(stat) return -1;
        stat = RETURN_ERROR(cudaMalloc((void**)&d_b, S * sizeof(T)));
        if(stat) return -1;
        stat = RETURN_ERROR(cudaMalloc((void**)&d_c, S * sizeof(T)));
        if(stat) return -1;

        stat = RETURN_ERROR(cudaMemcpy(d_a, a, S, cudaMemcpyHostToDevice));
        if(stat) return -1;
        stat = RETURN_ERROR(cudaMemcpy(d_b, b, S, cudaMemcpyHostToDevice));
        if(stat) return -1;

        // load one block of kernel with S threads, S must no more than limitation( typically 512 )
        gpu_add_kernel_v1<T, S><<<1, S>>>(d_a, d_b, d_c);

        stat = RETURN_ERROR(cudaMemcpy(c, d_c, S, cudaMemcpyDeviceToHost));
        if(stat) return -1;

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        return stat;
    }

    template<typename T, size_t S>
    int gpu_add_v2(T a[S], T b[S], T c[S]) {
        std::cout << "GPU vec add v2 for pointer-like array" << std::endl;
        // Choose limit less than the maximum limit
        constexpr int thread_limit = 128;
        constexpr int element_max = 1024;
        int stat = 0;
        T *d_a, *d_b, *d_c;

        stat = RETURN_ERROR(cudaMalloc((void**)&d_a, S * sizeof(T)));
        if(stat) return -1;
        stat = RETURN_ERROR(cudaMalloc((void**)&d_b, S * sizeof(T)));
        if(stat) return -1;
        stat = RETURN_ERROR(cudaMalloc((void**)&d_c, S * sizeof(T)));
        if(stat) return -1;

        stat = RETURN_ERROR(cudaMemcpy(d_a, a, S, cudaMemcpyHostToDevice));
        if(stat) return -1;
        stat = RETURN_ERROR(cudaMemcpy(d_b, b, S, cudaMemcpyHostToDevice));
        if(stat) return -1;

        dim3 grids = { (element_max + thread_limit - 1) / thread_limit };
        dim3 threads = { thread_limit };
        gpu_add_kernel_v2<T, S><<<grids, threads>>>(d_a, d_b, d_c);

        stat = RETURN_ERROR(cudaMemcpy(c, d_c, S, cudaMemcpyDeviceToHost));
        if(stat) return -1;

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        return stat;
    }

    template<typename T, size_t S>
    int gpu_add(T (&a)[S], T (&b)[S], T (&c)[S]) {
        std::cout << "GPU vec add for c-style array" << std::endl;
        return gpu_add<T, S>(a, b, c);
    }

    int vec_add(void) {
        using std::vector;
        constexpr size_t N = 1024;
        [[maybe_unused]] constexpr size_t hardware_thread_limitation = 1024;
        [[maybe_unused]] constexpr size_t hardware_block_limitation = 2147363847;
        int stat = 0;
        int *c_a = new int[N];
        int *c_b = new int[N];
        int *c_c = new int[N];
        int *c_c_v1 = new int[N];
        int *c_c_v2 = new int[N];

        for(int i = 0; i < N; ++ i) {
            c_a[i] = -i;
            c_b[i] = i * i;
        }

        auto start = std::chrono::high_resolution_clock::now();
        gpu_add<int, N>(c_a, c_b, c_c);
        auto stp1 = std::chrono::high_resolution_clock::now();
        if(N <= hardware_thread_limitation) {
            // Should fail when N > 1024
            gpu_add_v1<int, N>(c_a, c_b, c_c_v1);
        } else {
            // Should apply to aribitrary N
            gpu_add_v2<int, N>(c_a, c_b, c_c_v2);
        }
        auto stp2 = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < N; ++ i) {
            bool ok = false;
            if(N <= hardware_thread_limitation) {
                ok = (c_c[i] == c_c_v1[i]);
            } else {
                ok = (c_c[i] == c_c_v2[i]);
            }
            if(!ok) {
                std::cout << std::endl << "Vector add error at(:" << i << ")!" << std::endl;
                return -1;
            }
        }

        auto dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(stp1 - start);
        auto dur2 = std::chrono::duration_cast<std::chrono::milliseconds>(stp2 - stp1);

        std::cout << "GPU vec add version 1 took time:" << dur1.count() << std::endl;
        std::cout << "GPU vec add version " << ((N <= hardware_thread_limitation) ? 2 : 3) << "(" << N << ") took time:" << dur2.count() << std::endl;

        delete[] c_a;
        delete[] c_b;
        delete[] c_c;
        delete[] c_c_v1;
        delete[] c_c_v2;

        return stat;
    }

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

        float *a, *b, sum = 0, *partial_sum;
        float *dev_a, *dev_b, *dev_partial_sum;

        // CPU mem alloc
        a = new float[N];
        b = new float[N];
        partial_sum = new float[blocksPerGrid];
        // GPU mem alloc
        stat = RETURN_ERROR( cudaMalloc( (void**)&dev_a, N*sizeof(float) ) );
        if(stat) return -1;
        stat = RETURN_ERROR( cudaMalloc( (void**)&dev_b, N*sizeof(float) ) );
        if(stat) return -1;
        stat = RETURN_ERROR( cudaMalloc( (void**)&dev_partial_sum, blocksPerGrid*sizeof(float) ) );
        if(stat) return -1;

        for (int i = 0; i < N; ++ i) {
            a[i] = i;
            b[i] = i * 2;
        }

        stat = RETURN_ERROR( cudaMemcpy( dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice ) );
        if(stat) return -1;
        stat = RETURN_ERROR( cudaMemcpy( dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice ) );
        if(stat) return -1;

        gpu_dot<threadsPerBlock, N><<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_sum);

        stat = RETURN_ERROR( cudaMemcpy( partial_sum, dev_partial_sum, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost ) );
        if(stat) return -1;

        for(int i = 0; i < blocksPerGrid; ++i) {
            sum += partial_sum[i];
        }

        // fixme: cleanup should be done before return!
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_partial_sum);

        delete[] a;
        delete[] b;
        delete[] partial_sum;
        
        std::cout << "GPU vec dot product is: " << sum << std::endl;

        return stat;
    }

    int gpu_ripple() {
        constexpr int DIM = 16 << 6;
        int stat = 0;
        DataBlock<DIM> data;
        CPUAnimBitmap bitmap(DIM, DIM, &data);
        data.bitmap = &bitmap;

        stat = RETURN_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));
        if(stat) return -1;

        bitmap.anim_and_exit( (void (*)(void*, int))DataBlock<DIM>::generate_frame, (void (*)(void*))DataBlock<DIM>::clean_up);

        return stat;
    };

    int ch_5(void) {
        int stat = 0;

        stat = vec_add();
        if(stat) return -1;

        stat = vec_dot();
        if(stat) return -1;

        stat = gpu_ripple();
        if(stat) return -1;

        return stat;
    }
}