# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda
#Location of Alpaka and Cupla Toolkit
CUPLA_ROOT := /data/user/wredjeb/cupla
EIGEN_ROOT := /data/cmssw/slc7_amd64_gcc820/external/eigen/e4c107b451c52c9ab2d7b7fa4194ee35332916ec-pafccj/include/eigen3
TBB_ROOT := /data/cmssw/slc7_amd64_gcc820/external/tbb/2019_U3-pafccj


# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)

# operating system
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
TARGET_SIZE := 64
NVCCFLAGS   := -m${TARGET_SIZE} -w -lineinfo --expt-extended-lambda --use_fast_math --ftz=false --cudart shared
CCFLAGS     := -std=c++14 -O2 -g -I/data/cmssw/slc7_amd64_gcc820/external/boost/1.67.0-pafccj/include/ 
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS := --expt-relaxed-constexpr
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))


#TBB_BASE := /data/cmssw/slc7_amd64_gcc820/external/tbb/2019_U8/
# Common includes and paths for CUDA
INCLUDES   := -I../../common/inc -I include -I $(CUPLA_ROOT)/include -I $(CUPLA_ROOT)/alpaka/include -I $(EIGEN_ROOT) -I $(TBB_ROOT)/include
LIBRARIES  :=
CUPLA_CUDA_ACC := -DFOR_CUDA
CUPLA_CPUTBB_ACC := -I $(TBB_ROOT)/include -L $(TBB_ROOT)/lib -ltbb

CUDA_FLAGS := -x cu
CUPLA_FLAGS := -DUSE_CUPLA

################################################################################

# Gencode arguments
SMS ?= 60 70 75

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -DALPAKA_CUDA_ARCH=60:70:75
endif

################################################################################

# Target rules
all: build clean #need to specify clean in all.

build: mainAlpaka 

testSerAlpaka.o:src/testSerAlpaka.cc
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

testTBBAlpaka.o:src/testTBBAlpaka.cc
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(CUPLA_CPUTBB_ACC) -o $@ -c $<

testGpuAlpaka.o:src/testGpuAlpaka.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mainAlpaka.o:src/mainAlpaka.cc
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mainAlpaka: mainAlpaka.o testSerAlpaka.o testTBBAlpaka.o testGpuAlpaka.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(CUPLA_CPUTBB_ACC) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) mainAlpaka


clean:
	rm -f mainAlpaka.o testSerAlpaka.o testTBBAlpaka.o testGpuAlpaka.o


clobber: clean
