#include <algorithm>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <cpu_anim.h>
#include <book_interface.h>
#include <book.h>

namespace ch7 {
    static texture<float> inSrcRef1D;
    static texture<float> outSrcRef1D;
    static texture<float> constSrcRef1D;
    static texture<float, 2> inSrcRef2D;
    static texture<float, 2> outSrcRef2D;
    static texture<float, 2> constSrcRef2D;

    class LabWarpper {
    public:
        constexpr static int DIM = 512;
        constexpr static int BLK = 16;
        constexpr static float MIN_TAM = .0001f;
        constexpr static float MAX_TAM = 1.f;
        constexpr static float TRANS_RATE = 0.20f;
        constexpr static float BIAS = 1e-10f;
        enum class MemoryPattern {
            Access1D,
            Access2D,
            AccessGlb,
        };
    private:
        CPUAnimBitmap *bitmap {nullptr};
        cudaEvent_t start, stp1, stp2, end;
        float *inSrcGlobal {nullptr}, *inSrc1D {nullptr}, *inSrc2D {nullptr};
        float *outSrcGlobal {nullptr}, *outSrc1D {nullptr}, *outSrc2D {nullptr};
        float *constSrcGlobal {nullptr}, *constSrc1D {nullptr}, *constSrc2D {nullptr};
        size_t inSrcPitch = 0, outSrcPitch = 0, constSrcPitch = 0;
        unsigned char *devBitmap {nullptr};
        int stat {0};
        float times[3] {0};
        long long frames[3] {0};
    protected:
        void const_gen(float *constSrc) {
            for(int x = 0; x < DIM; ++ x) {
                for(int y= 0; y < DIM; ++ y) {
                    int offset = y * DIM + x;
                    if(x >= DIM / 4 && y >= DIM / 4 && x <= DIM / 4 + 20 && y <= DIM / 4 + 20) {
                        constSrc[offset] = MAX_TAM;
                    } else if(x >= DIM / 2 && y >= DIM / 2 && x <= DIM / 2 + 10 && y <= DIM / 2 + 10) {
                        constSrc[offset] = MIN_TAM;
                    } else if(x >= DIM / 4 * 3 && y >= DIM / 4 * 3 && x <= DIM / 4 * 3 + 30 && y <= DIM / 4 * 3 + 30) {
                        constSrc[offset] = MAX_TAM;
                    } else {
                        constSrc[offset] = 0.0f;
                    }
                }
            }
        }
        void clean_up() {
            delete bitmap;

            HANDLE_ERROR(cudaEventDestroy(start));
            HANDLE_ERROR(cudaEventDestroy(stp1));
            HANDLE_ERROR(cudaEventDestroy(stp2));
            HANDLE_ERROR(cudaEventDestroy(end));

            cudaFree(inSrcGlobal);
            cudaFree(outSrcGlobal);
            cudaFree(constSrcGlobal);
            HANDLE_ERROR(cudaUnbindTexture(inSrcRef1D));
            HANDLE_ERROR(cudaUnbindTexture(outSrcRef1D));
            HANDLE_ERROR(cudaUnbindTexture(constSrcRef1D));
            cudaFree(inSrc1D);
            cudaFree(outSrc1D);
            cudaFree(constSrc1D);
            HANDLE_ERROR(cudaUnbindTexture(inSrcRef2D));
            HANDLE_ERROR(cudaUnbindTexture(outSrcRef2D));
            HANDLE_ERROR(cudaUnbindTexture(constSrcRef2D));
            cudaFree(inSrc2D);
            cudaFree(outSrc2D);
            cudaFree(constSrc2D);
            cudaFree(devBitmap);
        }
        void set_up() {
            auto desc1D = cudaCreateChannelDesc<float>();
            auto desc2D = cudaCreateChannelDesc<float>();
            auto sz = DIM * DIM * sizeof(float);
            bitmap = new CPUAnimBitmap { DIM, DIM, this };

            if(stat = RETURN_ERROR( cudaMalloc((void**)&devBitmap, sz) ))
                throw std::runtime_error("cudaMalloc devBitmap");

            if(stat = RETURN_ERROR(cudaEventCreate(&start)))
                throw std::runtime_error("cudaEventCreate start");
            if(stat = RETURN_ERROR(cudaEventCreate(&stp1)))
                throw std::runtime_error("cudaEventCreate stp1");
            if(stat = RETURN_ERROR(cudaEventCreate(&stp2)))
                throw std::runtime_error("cudaEventCreate stp2");
            if(stat = RETURN_ERROR(cudaEventCreate(&end)))
                throw std::runtime_error("cudaEventCreate end");

            if(stat = RETURN_ERROR( cudaMalloc((void**)&inSrcGlobal, sz) ))
                throw std::runtime_error("cudaMalloc inSrc");
            if(stat = RETURN_ERROR( cudaMalloc((void**)&outSrcGlobal, sz) ))
                throw std::runtime_error("cudaMalloc outSrc");
            if(stat = RETURN_ERROR( cudaMalloc((void**)&constSrcGlobal, sz) ))
                throw std::runtime_error("cudaMalloc constSrc");

            if(stat = RETURN_ERROR( cudaMalloc((void**)&inSrc1D, sz) ))
                throw std::runtime_error("cudaMalloc inSrc");
            if(stat = RETURN_ERROR( cudaMalloc((void**)&outSrc1D, sz) ))
                throw std::runtime_error("cudaMalloc outSrc");
            if(stat = RETURN_ERROR( cudaMalloc((void**)&constSrc1D, sz) ))
                throw std::runtime_error("cudaMalloc constSrc");

            if(stat = RETURN_ERROR( cudaMallocPitch((void**)&inSrc2D, &inSrcPitch, DIM * sizeof(float), DIM) ))
                throw std::runtime_error("cudaMalloc inSrc2D");
            if(stat = RETURN_ERROR( cudaMallocPitch((void**)&outSrc2D, &outSrcPitch, DIM * sizeof(float), DIM) ))
                throw std::runtime_error("cudaMalloc outSrc2D");
            if(stat = RETURN_ERROR( cudaMallocPitch((void**)&constSrc2D, &constSrcPitch, DIM * sizeof(float), DIM) ))
                throw std::runtime_error("cudaMalloc constSrc2D");

            if(stat = RETURN_ERROR( cudaBindTexture(0, inSrcRef1D, inSrc1D, desc1D, sz) ))
                throw std::runtime_error("cudaBindTexture inSrcRef1d");
            if(stat = RETURN_ERROR( cudaBindTexture(0, outSrcRef1D, outSrc1D, desc1D, sz) ))
                throw std::runtime_error("cudaBindTexture outSrc");
            if(stat = RETURN_ERROR( cudaBindTexture(0, constSrcRef1D, constSrc1D, desc1D, sz) ))
                throw std::runtime_error("cudaBindTexture constSrc");

            if(stat = RETURN_ERROR( cudaBindTexture2D(0, inSrcRef2D, inSrc2D, desc2D, DIM, DIM, inSrcPitch) ))
                throw std::runtime_error("cudaBindTexture inSrcRef2d");
            if(stat = RETURN_ERROR( cudaBindTexture2D(0, outSrcRef2D, outSrc2D, desc2D, DIM, DIM, outSrcPitch) ))
                throw std::runtime_error("cudaBindTexture outSrcRef2D");
            if(stat = RETURN_ERROR( cudaBindTexture2D(0, constSrcRef2D, constSrc2D, desc2D, DIM, DIM, constSrcPitch) ))
                throw std::runtime_error("cudaBindTexture constSrcRef2D");

        }
    public:
        static void anim_gen(LabWarpper *d, int ticks);
        static void anim_clean(LabWarpper *d);
        LabWarpper(){
            set_up();
        }
        ~LabWarpper(){
            clean_up();
        }
        int operator()();
        CPUAnimBitmap* get_bitmap() {
            return bitmap;
        }
    };

    namespace GPUWarpper {
        template<int choose>
        __global__ static void copy_const_kernel(float *devPtr, float *devIn = nullptr) {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int offset = x + y * blockDim.x * gridDim.x;

            float tmp = 0.f;
            switch (choose)
            {
            case static_cast<int>(LabWarpper::MemoryPattern::Access1D):
                tmp = tex1Dfetch(constSrcRef1D, offset);
                break;
            case static_cast<int>(LabWarpper::MemoryPattern::Access2D):
                tmp = tex2D(constSrcRef2D, x, y);
                break;
            case static_cast<int>(LabWarpper::MemoryPattern::AccessGlb):
                if(devIn)
                    tmp = devIn[offset];
                break;
            default:
                break;
            }
            if(tmp > LabWarpper::BIAS) {
                devPtr[offset] = tmp;
            }
        }

        template<int choose>
        __global__ static void blend_kernel(float* devPtr, int dataIn, float* devIn = nullptr) {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int offset = x + y * blockDim.x * gridDim.x;
            int upperLimit = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
            int lowerLimit = 0;
            int mod = blockDim.x * gridDim.x;

            float ans = 0.f, center = 0.f;

            int top = offset + blockDim.x * gridDim.x;
            if(top >= upperLimit) top -= blockDim.x * gridDim.x;
            int down = offset - blockDim.x * gridDim.x;
            if(down < lowerLimit) down += blockDim.x * gridDim.x;
            int right = offset + 1;
            if(right % mod < offset % mod) right -= 1;
            int left = offset - 1;
            if(left < lowerLimit || left % mod > offset % mod) left += 1;

            switch (choose)
            {
            case static_cast<int>(LabWarpper::MemoryPattern::Access1D):
                if(dataIn) {
                    center = tex1Dfetch(inSrcRef1D, offset);
                    ans += tex1Dfetch(inSrcRef1D, top);
                    ans += tex1Dfetch(inSrcRef1D, down);
                    ans += tex1Dfetch(inSrcRef1D, left);
                    ans += tex1Dfetch(inSrcRef1D, right);
                } else {
                    center = tex1Dfetch(outSrcRef1D, offset);
                    ans += tex1Dfetch(outSrcRef1D, top);
                    ans += tex1Dfetch(outSrcRef1D, down);
                    ans += tex1Dfetch(outSrcRef1D, left);
                    ans += tex1Dfetch(outSrcRef1D, right);
                }
                break;
            case static_cast<int>(LabWarpper::MemoryPattern::Access2D):
                if(dataIn) {
                    center = tex2D(inSrcRef2D, x, y);
                    ans += tex2D(inSrcRef2D, x, y + 1);
                    ans += tex2D(inSrcRef2D, x, y - 1);
                    ans += tex2D(inSrcRef2D, x - 1, y);
                    ans += tex2D(inSrcRef2D, x + 1, y);
                } else {
                    center = tex2D(outSrcRef2D, x, y);
                    ans += tex2D(outSrcRef2D, x, y + 1);
                    ans += tex2D(outSrcRef2D, x, y - 1);
                    ans += tex2D(outSrcRef2D, x - 1, y);
                    ans += tex2D(outSrcRef2D, x + 1, y);
                }
                break;
            case static_cast<int>(LabWarpper::MemoryPattern::AccessGlb):
                if(devIn) {
                    center = devIn[offset];
                    ans += devIn[top];
                    ans += devIn[down];
                    ans += devIn[left];
                    ans += devIn[right];
                }
                break;
            default:
                break;
            }
            ans = center + LabWarpper::TRANS_RATE * (ans - 4 * center);
            devPtr[offset] = ans;
        }
    }

    void LabWarpper::anim_clean(LabWarpper *d) {
        d->clean_up();
    }

    void LabWarpper::anim_gen(LabWarpper *d, int ticks) {
        constexpr int LOOP = 90;
        dim3 grid { DIM / BLK, DIM / BLK };
        dim3 blk { BLK, BLK };

        int dataIn = 1;
        cudaEventRecord(d->start, 0);
        // Global memory fetch
        for(int i = 0; i < LOOP; ++ i) {
            GPUWarpper::copy_const_kernel<static_cast<int>(LabWarpper::MemoryPattern::AccessGlb)><<<grid, blk>>>(d->inSrcGlobal, d->constSrcGlobal);
            GPUWarpper::blend_kernel<static_cast<int>(LabWarpper::MemoryPattern::AccessGlb)><<<grid, blk>>>(d->outSrcGlobal, dataIn, d->inSrcGlobal);
            std::swap(d->inSrcGlobal, d->outSrcGlobal);
            dataIn = !dataIn;
        }
        cudaEventRecord(d->stp1, 0);
        cudaEventSynchronize(d->stp1);
        // Texture 2d fetch
        for(int i = 0; i < LOOP; ++ i) {
            GPUWarpper::copy_const_kernel<static_cast<int>(LabWarpper::MemoryPattern::Access2D)><<<grid, blk>>>(d->inSrc2D);
            GPUWarpper::blend_kernel<static_cast<int>(LabWarpper::MemoryPattern::Access2D)><<<grid, blk>>>(d->outSrc2D, dataIn);
            std::swap(d->inSrc2D, d->outSrc2D);
            dataIn = !dataIn;
        }
        cudaEventRecord(d->stp2, 0);
        cudaEventSynchronize(d->stp2);
        // Texture 1d fetch
        for(int i = 0; i < LOOP; ++ i) {
            GPUWarpper::copy_const_kernel<static_cast<int>(LabWarpper::MemoryPattern::Access1D)><<<grid, blk>>>(d->inSrc1D);
            GPUWarpper::blend_kernel<static_cast<int>(LabWarpper::MemoryPattern::Access1D)><<<grid, blk>>>(d->outSrc1D, dataIn);
            std::swap(d->inSrc1D, d->outSrc1D);
            dataIn = !dataIn;
        }
        cudaEventRecord(d->end, 0);
        cudaEventSynchronize(d->end);

        float_to_color<<<grid, blk>>>(d->devBitmap, d->inSrc1D);

        float tmp = 0;
        cudaEventElapsedTime(&tmp, d->start, d->stp1);
        d->times[static_cast<int>(LabWarpper::MemoryPattern::Access1D)] += tmp;
        d->frames[static_cast<int>(LabWarpper::MemoryPattern::Access1D)] += 1;
        cudaEventElapsedTime(&tmp, d->stp1, d->stp2);
        d->times[static_cast<int>(LabWarpper::MemoryPattern::Access2D)] += tmp;
        d->frames[static_cast<int>(LabWarpper::MemoryPattern::Access2D)] += 1;
        cudaEventElapsedTime(&tmp, d->stp2, d->end);
        d->times[static_cast<int>(LabWarpper::MemoryPattern::AccessGlb)] += tmp;
        d->frames[static_cast<int>(LabWarpper::MemoryPattern::AccessGlb)] += 1;

        float a1d = d->times[static_cast<int>(LabWarpper::MemoryPattern::Access1D)] / d->frames[static_cast<int>(LabWarpper::MemoryPattern::Access1D)],
            a2d = d->times[static_cast<int>(LabWarpper::MemoryPattern::Access2D)] / d->frames[static_cast<int>(LabWarpper::MemoryPattern::Access2D)],
            agb = d->times[static_cast<int>(LabWarpper::MemoryPattern::AccessGlb)] / d->frames[static_cast<int>(LabWarpper::MemoryPattern::AccessGlb)];
        // Elapsed times are influenced by cache warmming up.
        std::cout << "Generate frame No." << d->frames[0] << ":" << std::endl;
        std::cout << "\tAverage time cost using 1d texture memory fetch: " << a1d << std::endl
            << "\tAverage time cost using 2d texture memory fetch: " << a2d << std::endl
            << "\tAverage time cost using global memory fetch: " << agb << std::endl;


        if(d->stat = RETURN_ERROR( cudaMemcpy(d->bitmap->get_ptr(), d->devBitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost)) )
            throw std::runtime_error("cudaMemcpy devBitmap");
    }

    int LabWarpper::operator()(){
        std::unique_ptr<float[]> tmp { new float[DIM * DIM] };

        const_gen(tmp.get());
        if(stat = RETURN_ERROR(cudaMemcpy(constSrc1D, tmp.get(), DIM * DIM * sizeof(float), cudaMemcpyHostToDevice)))
            return stat;
        if(stat = RETURN_ERROR(cudaMemcpy(constSrc2D, tmp.get(), DIM * DIM * sizeof(float), cudaMemcpyHostToDevice)))
            return stat;
        if(stat = RETURN_ERROR(cudaMemcpy(constSrcGlobal, tmp.get(), DIM * DIM * sizeof(float), cudaMemcpyHostToDevice)))
            return stat;

        this->get_bitmap()->anim_and_exit((void(*)(void*, int))anim_gen, (void(*)(void*))anim_clean);

        return stat;
    }

    int ch_7(){
        LabWarpper lab7;
        return lab7();
    }
}