#ifndef __BOOK_INTERFACE_H__
#define __BOOK_INTERFACE_H__
#include <cpu_anim.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

static int ReturnError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        return -1;
    }
    return 0;
}

#define RETURN_ERROR( err ) (ReturnError( err, __FILE__, __LINE__ ))

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

template<int DIM>
__global__ static void gpu_generate_frame(unsigned char *dev_bitmap, int ticks) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = gridDim.x * blockDim.x * y + x;

    float fx = x - (DIM / 2.f);
    float fy = y - (DIM / 2.f);
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char)(128.f + 127.f * cos(d / 10.f - ticks / 7.f) / (d / 10.f + 1.f));

    dev_bitmap[idx * 4 + 0] = grey;
    dev_bitmap[idx * 4 + 1] = grey;
    dev_bitmap[idx * 4 + 2] = grey;
    dev_bitmap[idx * 4 + 3] = 255;
}

template<int DIM>
struct DataBlock {
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
    // Clean up used device mem
    static void clean_up(unsigned char *dev_bitmap) {
        if(dev_bitmap) cudaFree( dev_bitmap );
    }
    // Generate frame used to animate
    static void generate_frame(DataBlock *d, int ticks) {
        constexpr int threads_size = 16;
        dim3 blocks (DIM / threads_size, DIM / threads_size);
        dim3 threads (threads_size, threads_size);

        gpu_generate_frame<DIM><<<blocks, threads>>>(d->dev_bitmap, ticks);

        HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
    }
    DataBlock(): dev_bitmap {nullptr}, bitmap (nullptr) {}
    ~DataBlock() {
        DataBlock::clean_up(this->dev_bitmap);
    }
};

struct cuComplex {
    float r;
    float i;
    cuComplex( float a, float b ) : r(a), i(b) {}
    float magnitude2( void ) { return r * r + i * i; }
    cuComplex operator*( const cuComplex& a ) {
        return cuComplex( r * a.r - i * a.i, i * a.r + r * a.i );
    }
    cuComplex operator+( const cuComplex& a ) {
        return cuComplex( r + a.r, i + a.i );
    }
};

struct cuComplexGPU {
    float r;
    float i;
    __device__ cuComplexGPU( float a, float b ) : r(a), i(b) {}
    __device__ float magnitude2( void ) { return r * r + i * i; }
    __device__ cuComplexGPU operator*( const cuComplexGPU& a ) {
        return cuComplexGPU( r * a.r - i * a.i, i * a.r + r * a.i );
    }
    __device__ cuComplexGPU operator+( const cuComplexGPU& a ) {
        return cuComplexGPU( r + a.r, i + a.i );
    }
};

#endif