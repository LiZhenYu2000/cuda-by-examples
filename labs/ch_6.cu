#include <iostream>
#include <cuda.h>
#include <book_interface.h>
#include <cpu_bitmap.h>

#define rnd( x ) (x * rand() / RAND_MAX)

namespace ch6 {
    constexpr int SPHERES = 50;
    __constant__ cuSphere spConst[SPHERES];

    template<int DIM, int BLK, int SPHERES>
    __global__ void gpu_raytracer(unsigned char *devBitmap, cuSphere *sps, const float INF) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = x + y * blockDim.x * gridDim.x;
        // Move zero point of cordinate to the middle of image
        float ox = x - DIM / 2.f;
        float oy = y - DIM / 2.f;
        // Set backgroud color of ray
        float r = 0, g = 0, b = 0;

        // Hit check for every sphere
        for(int i = 0; i < SPHERES; ++ i) {
            // Cos of angle between ray and normal vector of hit point
            float n;
            float dis = sps[i].hit(ox, oy, &n);
            if(dis > -INF) {
                // Reflection of ray
                float scale = n;
                r = sps[i].r * scale;
                g = sps[i].g * scale;
                b = sps[i].b * scale;
            }
        }

        // Set pixel color
        devBitmap[idx * 4 + 0] = (int)(r * 255);
        devBitmap[idx * 4 + 1] = (int)(g * 255);
        devBitmap[idx * 4 + 2] = (int)(b * 255);
        devBitmap[idx * 4 + 3] = 255;
    }

    template<int DIM, int BLK, int SPHERES>
    __global__ void gpu_raytracer_const(unsigned char *devBitmap, const float INF) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = x + y * blockDim.x * gridDim.x;
        // Move zero point of cordinate to the middle of image
        float ox = x - DIM / 2.f;
        float oy = y - DIM / 2.f;
        // Set backgroud color of ray
        float r = 0, g = 0, b = 0;

        // Hit check for every sphere
        for(int i = 0; i < SPHERES; ++ i) {
            // Cos of angle between ray and normal vector of hit point
            float n;
            float dis = spConst[i].hit(ox, oy, &n);
            if(dis > -INF) {
                // Reflection of ray
                float scale = n;
                r = spConst[i].r * scale;
                g = spConst[i].g * scale;
                b = spConst[i].b * scale;
            }
        }

        // Set pixel color
        devBitmap[idx * 4 + 0] = (int)(r * 255);
        devBitmap[idx * 4 + 1] = (int)(g * 255);
        devBitmap[idx * 4 + 2] = (int)(b * 255);
        devBitmap[idx * 4 + 3] = 255;
    }

    int ray_tracer() {
        constexpr int DIM = 1024;
        constexpr int BLK = 16;

        [[maybe_unused]] cuSphere* spGlobal = nullptr;
        [[maybe_unused]] cuSphere* spConstPtr = nullptr;
        cuSphere* spHost = nullptr;
        CPUBitmap bitmap { DIM, DIM };
        unsigned char* devBitmap;
        int stat = 0;

        cudaEvent_t start1, stop1, start2, stop2;
        auto cleanUp = [&stat, &start1, &start2, &stop1, &stop2, &spGlobal, &spHost, &devBitmap]()->int {
            HANDLE_ERROR( cudaEventDestroy(start1) );
            HANDLE_ERROR( cudaEventDestroy(start2) );
            HANDLE_ERROR( cudaEventDestroy(stop1) );
            HANDLE_ERROR( cudaEventDestroy(stop2) );
            delete[] spHost;
            cudaFree(spGlobal);
            cudaFree(devBitmap);
            return stat;
        };

        // CUDA event malloc
        stat = RETURN_ERROR( cudaEventCreate( &start1 ) );
        if(stat) return cleanUp();
        stat = RETURN_ERROR( cudaEventCreate( &start2 ) );
        if(stat) return cleanUp();
        stat = RETURN_ERROR( cudaEventCreate( &stop1 ) );
        if(stat) return cleanUp();
        stat = RETURN_ERROR( cudaEventCreate( &stop2 ) );
        if(stat) return cleanUp();
        // Bitmap malloc
        stat = RETURN_ERROR( cudaMalloc((void**)&devBitmap, bitmap.image_size()) );
        if(stat) return cleanUp();
        // Global device memory malloc
        stat = RETURN_ERROR( cudaMalloc((void**)&spGlobal, sizeof(cuSphere) * SPHERES) );
        if(stat) return cleanUp();
        // Host memory malloc
        spHost = new cuSphere[SPHERES];
        // Get constant memory symbol device address
        stat = RETURN_ERROR( cudaGetSymbolAddress((void**)&spConstPtr, spConst) );
        if(stat) return cleanUp();

        // Initialize host memory
        for(int i = 0; i < SPHERES; ++ i) {
            spHost[i].r = rnd(1.f);
            spHost[i].g = rnd(1.f);
            spHost[i].b = rnd(1.f);
            spHost[i].x = rnd( 1000.f ) - 500;
            spHost[i].y = rnd( 1000.f ) - 500;
            spHost[i].z = rnd( 1000.f ) - 500;
            spHost[i].rad = rnd( 100.f ) + 20;
        }

        // Memcpy to device global memory
        stat = RETURN_ERROR( cudaMemcpy(spGlobal, spHost, sizeof(cuSphere) * SPHERES, cudaMemcpyHostToDevice) );
        if(stat) return cleanUp();
        // Memcpy to device constant memory
        stat = RETURN_ERROR( cudaMemcpyToSymbol(spConst, spHost, sizeof(cuSphere) * SPHERES) );
        if(stat) return cleanUp();

        dim3 grid { DIM / BLK, DIM / BLK };
        dim3 blk { BLK, BLK };
        // CUDA event start
        cudaEventRecord(start1, 0);
        // Ray trace using global memory
        gpu_raytracer<DIM, BLK, SPHERES><<<grid, blk>>>(devBitmap, spGlobal, cuSphere::INF);
        // CUDA event stop1
        cudaEventRecord(stop1, 0);
        cudaEventSynchronize(stop1);

        cudaEventRecord(start2, 0);
        // Ray trace using constant memory
        // Use spConstPtr after get symbol address on device
        gpu_raytracer<DIM, BLK, SPHERES><<<grid, blk>>>(devBitmap, spConstPtr, cuSphere::INF);
        // gpu_raytracer_const<DIM, BLK, SPHERES><<<grid, blk>>>(devBitmap, cuSphere::INF);
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);

        // Calculate elapsed time
        float elapsedTime = 0;
        stat = RETURN_ERROR( cudaEventElapsedTime(&elapsedTime, start1, stop1) );
        if(stat) return cleanUp();
        std::cout << "Time to generate image using global memory: " << std::cout.precision(6) << elapsedTime << std::endl;
        stat = RETURN_ERROR( cudaEventElapsedTime(&elapsedTime, start2, stop2) );
        if(stat) return cleanUp();
        std::cout << "Time to generate image using constant memory: " << std::cout.precision(6) << elapsedTime << std::endl;

        // Memcpy to host bitmap
        stat = RETURN_ERROR( cudaMemcpy(bitmap.get_ptr(), devBitmap, bitmap.image_size(), cudaMemcpyDeviceToHost) );
        if(stat) return cleanUp();

        // Draw image
        bitmap.display_and_exit();

        return cleanUp();
    }

    int ch_6() {
        int stat = 0;

        stat = ray_tracer();
        if(stat) return -1;

        return stat;
    }
}