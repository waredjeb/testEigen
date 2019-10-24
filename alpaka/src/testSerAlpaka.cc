
#define CMS_ARCHITECTURE CPU_SERIAL

#include <alpaka/standalone/CpuSerial.hpp>
#include <alpaka/alpaka.hpp>
using Dim = alpaka::dim::DimInt<1u>;
using Idx = std::size_t;
using Acc = alpaka::acc::AccCpuSerial<Dim,Idx>;
using DevAcc = alpaka::dev::Dev<Acc>;
using PltfAcc = alpaka::pltf::Pltf<DevAcc>; //Platform specific
using QueueAcc = alpaka::queue::QueueCpuSync;
#include "testEigenAlpakaNoFit.cc"