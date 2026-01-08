#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <book_interface.hpp>
#include <book.h>
#include <cpu_bitmap.h>
#include <thread>

template<typename T, size_t S>
int cpu_add(T a[S], T b[S], T c[S]) {
    std::cout << "CPU vec add for pointer-like array" << std::endl;
    for(size_t i = 0; i < S; ++ i) {
        c[i] = a[i] + b[i];
    }
    return 0;
}

template<typename T, size_t S>
int cpu_add(T (&a)[S], T (&b)[S], T (&c)[S]) {
    std::cout << "CPU vec add for c-style array" << std::endl;
    T *ta = a, *tb = b, *tc = c;
    return cpu_add<T, S>(ta, tb, tc);
}

template<typename T>
int cpu_add(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c) {
    std::cout << "CPU vec add for cpp-style vector" << std::endl;
    size_t sz = std::min({a.size(), b.size(), c.size()});
    for(size_t i = 0; i < sz; ++ i) {
        c[i] = a[i] + b[i];
    }
    return 0;
}

template<typename T, size_t S>
__global__ void gpu_add_kernel(T *a, T *b, T *c) {
    size_t tid = blockIdx.x;
    if (tid < S)
        c[tid] = a[tid] + b[tid];
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
int gpu_add(T (&a)[S], T (&b)[S], T (&c)[S]) {
    std::cout << "GPU vec add for c-style array" << std::endl;
    return gpu_add<T, S>(a, b, c);
}

int vec_add(void) {
    using std::vector;
    constexpr size_t N = 1000;
    int stat = 0;
    int *c_a = new int[N];
    int *c_b = new int[N];
    int *c_c = new int[N];
    vector<int> v_a(c_a, c_a + N),
        v_b(c_b, c_b + N),
        v_c(c_c, c_c + N);

    for(int i = 0; i < N; ++ i) {
        v_a[i] = c_a[i] = -i;
        v_b[i] = c_b[i] = i * i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cpu_add(v_a, v_b, v_c);
    auto stp1 = std::chrono::high_resolution_clock::now();
    cpu_add<int, N>(c_a, c_b, c_c);
    auto stp2 = std::chrono::high_resolution_clock::now();
    gpu_add<int, N>(c_a, c_b, c_c);
    auto stp3 = std::chrono::high_resolution_clock::now();

    auto dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(stp1 - start);
    auto dur2 = std::chrono::duration_cast<std::chrono::milliseconds>(stp2 - stp1);
    auto dur3 = std::chrono::duration_cast<std::chrono::milliseconds>(stp3 - stp2);

    std::cout << "CPU cpp-style vector add took time: " << dur1.count() << std::endl;
    std::cout << "CPU c-style vector add took time: " << dur2.count() << std::endl;
    std::cout << "GPU c-style vector add took time: " << dur3.count() << std::endl;

    delete[] c_a;
    delete[] c_b;
    delete[] c_c;

    return stat;
}

template<size_t DIM>
__device__ bool gpu_julia_set_check(int x, int y) {
    constexpr float scale = 1.5;
    constexpr size_t iterations = 200;
    constexpr float threshold = 1000;
    // Get complex coordinate for each point
    // Buggy(size_t DIM could overflow):
    // float jx = scale * (float) (DIM / 2 - x) / (DIM / 2);
    // float jy = scale * (float) (DIM / 2 - y) / (DIM / 2);
    // OK:
    float jx = scale * (DIM / 2. - x) / (DIM / 2.);
    float jy = scale * (DIM / 2. - y) / (DIM / 2.);

    cuComplexGPU c(-.8, +.156);
    cuComplexGPU a(jx, jy);

    for (int i = 0; i < iterations; ++ i) {
        a = a * a + c;
        if( a.magnitude2() > threshold ) {
            return false;
        }
    }
    return true;
}

template<size_t DIM>
bool cpu_julia_set_check(int x, int y) {
    constexpr float scale = 1.5;
    constexpr size_t iterations = 200;
    constexpr float threshold = 1000;
    // Get complex coordinate for each point
    // Buggy(size_t DIM could overflow):
    // float jx = scale * (float) (DIM / 2 - x) / (DIM / 2);
    // float jy = scale * (float) (DIM / 2 - y) / (DIM / 2);
    float jx = scale * (DIM / 2. - x) / (DIM / 2.);
    float jy = scale * (DIM / 2. - y) / (DIM / 2.);

    cuComplex c(-.8, +.156);
    cuComplex a(jx, jy);

    for (int i = 0; i < iterations; ++ i) {
        a = a * a + c;
        if( a.magnitude2() > threshold ) {
            return false;
        }
    }
    return true;
}

template<size_t DIM>
void cpu_julia_set_kernel(unsigned char *ptr) {
    for(int y = 0; y < DIM; ++ y) {
        for(int x = 0; x < DIM; ++ x) {
            int offset = x + y * DIM;
            bool isJulia = cpu_julia_set_check<DIM>(x, y);

            ptr[offset * 4 + 0] = 255 * (int)isJulia;
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 255;
        }
    }
}

template<size_t DIM>
__global__ void gpu_julia_set_kernel(unsigned char *ptr) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    bool isJulia = gpu_julia_set_check<DIM>( x, y );

    ptr[offset * 4 + 0] = 255 * (int)isJulia;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

template<size_t DIM = 400>
int julia_set(void) {
    int stat = 0;
    CPUBitmap cpu_bitmap { DIM, DIM, "cpu_bitmap" };
    CPUBitmap gpu_bitmap { DIM, DIM, "gpu_bitmap" };
    unsigned char *dev_ptr = nullptr;
    dim3 grid = { DIM, DIM };
    unsigned char *c_ptr = cpu_bitmap.get_ptr();
    unsigned char *g_ptr = gpu_bitmap.get_ptr();

    stat = RETURN_ERROR( cudaMalloc((void**)&dev_ptr, gpu_bitmap.image_size()) );
    if(stat) return -1;

    auto start = std::chrono::high_resolution_clock::now();

    cpu_julia_set_kernel<DIM>( c_ptr );

    auto stp1 = std::chrono::high_resolution_clock::now();

    gpu_julia_set_kernel<DIM><<<grid, 1>>>( dev_ptr );

    stat = RETURN_ERROR( cudaMemcpy(g_ptr, dev_ptr, gpu_bitmap.image_size(), cudaMemcpyDeviceToHost) );
    if(stat) return -1;

    cudaFree(dev_ptr);

    auto stp2 = std::chrono::high_resolution_clock::now();

    auto dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(stp1 - start);
    auto dur2 = std::chrono::duration_cast<std::chrono::milliseconds>(stp2 - stp1);

    std::cout << "CPU julia calculate took time: " << dur1.count() << std::endl;
    std::cout << "GPU julia calculate took time: " << dur2.count() << std::endl;

    // cpu_bitmap.display_and_exit();
    gpu_bitmap.display_and_exit();

    return stat;
}

int ch_4(void) {
    int stat = 0;

    stat = vec_add();
    if(stat) return -1;

    stat = julia_set();
    if(stat) return -1;
    
    return stat;
}