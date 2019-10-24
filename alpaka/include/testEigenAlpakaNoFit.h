#ifndef TEST_EIGEN_CUPLA
#define TEST_EIGEN_CUPLA

#include <alpaka/alpaka.hpp>

namespace CPU_SERIAL {
  void testEigen();
}

namespace CPU_PARALLEL_TBB {
  void testEigen();
}

 namespace GPU_CUDA{
   void testEigen();
 }


#endif //TEST_EIGEN_CUPLA