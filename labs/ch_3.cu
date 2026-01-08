#include <iostream>
#include <book_interface.hpp>
#include <book_cpp.hpp>

__global__ void add(int a, int b, int* dev_c) {
    *dev_c = a + b;
}

int Intro(void) {
    int stat = 0;
    int a = 2, b = 3;
    int *host_c = nullptr, *dev_c = nullptr;

    // host mem alloc
    host_c = new int;
    if(!host_c) return -1;
    *host_c = 0;

    // dev mem alloc
    stat = RETURN_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
    if(stat) return -1;

    add<<<1, 1>>>(a, b, dev_c);
    RETURN_ERROR(cudaMemcpy(host_c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
    if(stat) return -1;

    cudaFree(dev_c);

    // print
    std::cout << "Intro(): " << *host_c << std::endl;
    return 0;
}

int QueryDevice(void) {
    int count = 0;
    int stat = RETURN_ERROR(cudaGetDeviceCount( &count ));
    if(stat) return -1;

    std::cout << "Querying Devices ..." << std::endl;

    // loop for all devs
    for(int i = 0; i < count; ++ i) {
        struct cudaDeviceProp prop;
        stat = RETURN_ERROR(cudaGetDeviceProperties(&prop, i));
        if(stat) return -1;
        std::cout << "Device " << i << " info:" << std::endl;
        std::cout << prop << std::endl;
    }

    return 0;
}

int ChooseDevice(void) {
    struct cudaDeviceProp prop;
    int dev = -1, stat = 0;

    std::cout << "Choosing Devices ..." << std::endl;

    // get current device id
    stat = RETURN_ERROR(cudaGetDevice( &dev ));
    if(stat) return -1;

    std::cout << "Current running on device: " << dev << "." << std::endl;

    // specify the revision to choose device
    memset(&prop, 0, sizeof(struct cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;
    stat = RETURN_ERROR(cudaChooseDevice(&dev, &prop));
    if(stat) return -1;

    std::cout << "The device which is closest to revision: " << dev << "." << std::endl;

    // set the device to run device code using dev id
    stat = RETURN_ERROR(cudaSetDevice(dev));
    if(stat) return -1;

    return 0;
}

int ch_3(void) {
    int stat = 0;

    stat = Intro();
    if(stat) return -1;

    stat = QueryDevice();
    if(stat) return -1;

    stat = ChooseDevice();
    if(stat) return -1;

    return 0;
}