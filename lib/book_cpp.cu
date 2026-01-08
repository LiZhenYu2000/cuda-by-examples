#include <iostream>
#include "book_cpp.hpp"

std::ostream& operator<<(std::ostream& out, struct cudaDeviceProp prop) {
    using std::cout;
    using std::endl;
    cout << "-- Device Info Begin --" << endl;
    cout << "-- Name: " << prop.name << endl;
    cout << "-- Compute capabality: " << prop.major << " " << prop.minor << endl;
    cout << "-- Clock rate: " << prop.clockRate << endl;
    cout << "-- Total global mem: " << prop.totalGlobalMem << endl;
    cout << "-- Total constant mem: " << prop.totalConstMem << endl;
    cout << "-- Shared mem per block: " << prop.sharedMemPerBlock << endl;
    cout << "-- Registers per block: " << prop.regsPerBlock << endl;
    cout << "-- Warp size: " << prop.warpSize << endl;
    cout << "-- Mem pitch: " << prop.memPitch << endl;
    cout << "-- Max threads per blocks: " << prop.maxThreadsPerBlock << endl;
    cout << "-- Max threads (x y z): " << prop.maxThreadsDim[0] << " " <<
        prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << endl;
    cout << "-- Max grid size (x y z): " << prop.maxGridSize[0] << " " <<
        prop.maxGridSize[1] << " " << prop.maxGridSize[2] << endl;
    cout << "-- Texture alignment: " << prop.textureAlignment << endl;
    cout << "-- Is overlap: " << bool(prop.deviceOverlap) << endl;
    cout << "-- Multiprocessor counts: " << prop.multiProcessorCount << endl;
    cout << "-- Kernel timeout enabled: " << prop.kernelExecTimeoutEnabled << endl;
    cout << "-- Is integrated: " << bool(prop.integrated) << endl;
    cout << "-- Can map host mem: " << bool(prop.canMapHostMemory) << endl;
    cout << "-- Compute mode: " << prop.computeMode << endl;
    cout << "-- Max texture 1d: " << prop.maxTexture1D << endl;
    cout << "-- Max texture 2d (x y): " << prop.maxTexture2D[0] << " " <<
        prop.maxTexture2D[1] << endl;
    cout << "-- Max texture 3d (x y z): " << prop.maxTexture3D[0] << " " <<
        prop.maxTexture3D[1] << " " << prop.maxTexture3D[2] << endl;
    cout << "-- Device Info End --";
    return out;
}