
#define CMS_ARCHITECTURE GPU_CUDA

#include <alpaka/standalone/GpuCudaRt.hpp>
#include <alpaka/alpaka.hpp>
using Dim = alpaka::dim::DimInt<1u>;
using Idx = std::size_t;
using Acc = alpaka::acc::AccGpuCudaRt<Dim,Idx>;
using DevAcc = alpaka::dev::Dev<Acc>;
using PltfAcc = alpaka::pltf::Pltf<DevAcc>; //Platform specific
using QueueAcc = alpaka::queue::QueueCudaRtAsync;
#include "testEigenAlpakaNoFit.cc"