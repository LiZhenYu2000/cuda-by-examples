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
        auto cleanUp = [&stat, &d_a, &d_b, &d_c](void)->int{
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            return stat;
        };

        stat = RETURN_ERROR(cudaMalloc((void**)&d_a, S * sizeof(T)));
        if(stat) return cleanUp();
        stat = RETURN_ERROR(cudaMalloc((void**)&d_b, S * sizeof(T)));
        if(stat) return cleanUp();
        stat = RETURN_ERROR(cudaMalloc((void**)&d_c, S * sizeof(T)));
        if(stat) return cleanUp();

        stat = RETURN_ERROR(cudaMemcpy(d_a, a, S, cudaMemcpyHostToDevice));
        if(stat) return cleanUp();
        stat = RETURN_ERROR(cudaMemcpy(d_b, b, S, cudaMemcpyHostToDevice));
        if(stat) return cleanUp();

        gpu_add_kernel<T, S><<<S, 1>>>(d_a, d_b, d_c);

        stat = RETURN_ERROR(cudaMemcpy(c, d_c, S, cudaMemcpyDeviceToHost));
        if(stat) return cleanUp();

        return cleanUp();
    }

    template<typename T, size_t S>
    int gpu_add_v1(T a[S], T b[S], T c[S]) {
        std::cout << "GPU vec add v1 for pointer-like array" << std::endl;
        int stat = 0;
        T *d_a, *d_b, *d_c;
        auto cleanUp = [&stat, &d_a, &d_b, &d_c](void)->int{
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            return stat;
        };

        stat = RETURN_ERROR(cudaMalloc((void**)&d_a, S * sizeof(T)));
        if(stat) return cleanUp();
        stat = RETURN_ERROR(cudaMalloc((void**)&d_b, S * sizeof(T)));
        if(stat) return cleanUp();
        stat = RETURN_ERROR(cudaMalloc((void**)&d_c, S * sizeof(T)));
        if(stat) return cleanUp();

        stat = RETURN_ERROR(cudaMemcpy(d_a, a, S, cudaMemcpyHostToDevice));
        if(stat) return cleanUp();
        stat = RETURN_ERROR(cudaMemcpy(d_b, b, S, cudaMemcpyHostToDevice));
        if(stat) return cleanUp();

        // load one block of kernel with S threads, S must no more than limitation( typically 512 )
        gpu_add_kernel_v1<T, S><<<1, S>>>(d_a, d_b, d_c);

        stat = RETURN_ERROR(cudaMemcpy(c, d_c, S, cudaMemcpyDeviceToHost));
        if(stat) return cleanUp();

        return cleanUp();
    }

    template<typename T, size_t S>
    int gpu_add_v2(T a[S], T b[S], T c[S]) {
        std::cout << "GPU vec add v2 for pointer-like array" << std::endl;
        // Choose limit less than the maximum limit
        constexpr int thread_limit = 128;
        constexpr int element_max = 1024;
        int stat = 0;
        T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        auto cleanUp = [&stat, &d_a, &d_b, &d_c](void)->int{
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            return stat;
        };

        stat = RETURN_ERROR(cudaMalloc((void**)&d_a, S * sizeof(T)));
        if(stat) return cleanUp();
        stat = RETURN_ERROR(cudaMalloc((void**)&d_b, S * sizeof(T)));
        if(stat) return cleanUp();
        stat = RETURN_ERROR(cudaMalloc((void**)&d_c, S * sizeof(T)));
        if(stat) return cleanUp();

        stat = RETURN_ERROR(cudaMemcpy(d_a, a, S, cudaMemcpyHostToDevice));
        if(stat) return cleanUp();
        stat = RETURN_ERROR(cudaMemcpy(d_b, b, S, cudaMemcpyHostToDevice));
        if(stat) return cleanUp();

        dim3 grids = { (element_max + thread_limit - 1) / thread_limit };
        dim3 threads = { thread_limit };
        gpu_add_kernel_v2<T, S><<<grids, threads>>>(d_a, d_b, d_c);

        stat = RETURN_ERROR(cudaMemcpy(c, d_c, S, cudaMemcpyDeviceToHost));
        if(stat) return cleanUp();

        return cleanUp();
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
        auto cleanUp = [&stat, &c_a, &c_b, &c_c, &c_c_v1, &c_c_v2]()->int{
            delete[] c_a;
            delete[] c_b;
            delete[] c_c;
            delete[] c_c_v1;
            delete[] c_c_v2;
            return stat;
        };

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
                return cleanUp();
            }
        }

        auto dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(stp1 - start);
        auto dur2 = std::chrono::duration_cast<std::chrono::milliseconds>(stp2 - stp1);

        std::cout << "GPU vec add version 1 took time:" << dur1.count() << std::endl;
        std::cout << "GPU vec add version " << ((N <= hardware_thread_limitation) ? 2 : 3) << "(" << N << ") took time:" << dur2.count() << std::endl;

        return cleanUp();
    }

    int gpu_ripple() {
        constexpr int DIM = 16 << 6;
        int stat = 0;
        DataBlock<DIM> data;
        CPUAnimBitmap bitmap(DIM, DIM, &data);
        data.bitmap = &bitmap;
        auto cleanUp = [&stat, &data](void)->int{
            cudaFree(data.dev_bitmap);
            return stat;
        };

        stat = RETURN_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));
        if(stat) return cleanUp();

        bitmap.anim_and_exit( (void (*)(void*, int))DataBlock<DIM>::generate_frame, (void (*)(void*))DataBlock<DIM>::clean_up);

        return cleanUp();
    };

    int ch_5_1(void) {
        int stat = 0;

        stat = vec_add();
        if(stat) return -1;

        stat = gpu_ripple();
        if(stat) return -1;

        return stat;
    }
}