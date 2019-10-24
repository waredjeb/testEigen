
#define CMS_ARCHITECTURE CPU_PARALLEL_TBB

#include <alpaka/standalone/CpuTbbBlocks.hpp>
#include <alpaka/alpaka.hpp>
using Dim = alpaka::dim::DimInt<1u>;
using Idx = std::size_t;
using Acc = alpaka::acc::AccCpuTbbBlocks<Dim, Idx>;
using DevAcc = alpaka::dev::Dev<Acc>;
using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
using QueueAcc = alpaka::queue::QueueCpuSync;
#include "testEigenAlpakaNoFit.cc"