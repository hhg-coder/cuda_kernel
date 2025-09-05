//
// Created by Tech on 25-6-10.
//

#ifndef COMMON_CUH
#define COMMON_CUH
#pragma once

#include <stdio.h>
#include <stdlib.h>


void setGPU() {
    int DeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&DeviceCount);
    if (error != cudaSuccess || DeviceCount == 0) {
        printf("CUDA Device not found\n");
        exit(-1);
    }
    else {
        printf("CUD Device found: %d\n", DeviceCount);
    }

    int DeviceID = 0;
    error = cudaSetDevice(DeviceID);
    if (error != cudaSuccess) {
        printf("fail to set GPU computing.\n");
        exit(-1);
    }
    else {
        printf("GPU computing.\n");
    }
}


#endif //COMMON_CUH
