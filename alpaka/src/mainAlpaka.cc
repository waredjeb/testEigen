#include <iostream>
#include "testEigenAlpakaNoFit.h"

int main() {

    printf("CPU SERIAL \n\n");
    CPU_SERIAL::testEigen();
    
    printf("\n\nCPU PARALLEL TBB\n\n ");
    CPU_PARALLEL_TBB::testEigen();

    printf("\n\n GPU CUDA\n\n");
    GPU_CUDA::testEigen();

    return 0;

}