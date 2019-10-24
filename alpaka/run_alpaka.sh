export CUDA_ROOT="/usr/local/cuda"
export ALPAKA_ROOT="/data/user/wredjeb/cupla/alpaka"
export CUPLA_ROOT="/data/user/wredjeb/cupla"

export SLC7_BASE="/data/cmssw/slc7_amd64_gcc820/external/boost/1.67.0-pafccj"
export TBB_BASE="/data/cmssw/slc7_amd64_gcc820/external/tbb/2019_U3-pafccj/"
export EIGEN_BASE="/data/cmssw/slc7_amd64_gcc820/external/eigen/e4c107b451c52c9ab2d7b7fa4194ee35332916ec-pafccj/include/eigen3"
export CXX="g++"
export CXXFLAGS="-m64 -std=c++14 -g -O2 -DALPAKA_DEBUG=0 -DALPAKA_CUDA_ARCH=60:70:75 -I$CUDA_ROOT/include -L$CUDA_ROOT/lib64 -lcudart -lcuda -I$ALPAKA_ROOT/include -I$CUPLA_ROOT/include -I$SLC7_BASE/include -I$EIGEN_BASE -I$TBB_BASE/include -L$TBB_BASE/lib -ltbb" 
export HOST_FLAGS="-fopenmp -pthread -fPIC -ftemplate-depth-512 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-attributes -Wno-reorder -Wno-sign-compare"

export NVCC="$CUDA_ROOT/bin/nvcc"
export NVCC_FLAGS="-ccbin $CXX -w -lineinfo --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math --ftz=false --cudart shared"


$CXX  $CXXFLAGS $HOST_FLAGS  mainAlpaka.cc -c 

$CXX  $CXXFLAGS $HOST_FLAGS testSerAlpaka.cc -c 

$CXX  $CXXFLAGS $HOST_FLAGS testTBBAlpaka.cc -c 

$NVCC $CXXFLAGS $NVCC_FLAGS testGpuAlpaka.cu -c 

$CXX $CXXFLAGS $HOST_FLAGS -o ./bin/mainAlpaka mainAlpaka.o testSerAlpaka.o testTBBAlpaka.o testGpuAlpaka.o

rm mainAlpaka.o testSerAlpaka.o testTBBAlpaka.o testGpuAlpaka.o

