#include <iostream>


#include <Eigen/Core>
#include <Eigen/Eigenvalues>


// #include <cuda_to_cupla.hpp>

using namespace Eigen;

using Matrix5d = Matrix<double, 8, 8>;

namespace CMS_ARCHITECTURE {

#include "test_common.h"

ALPAKA_FN_ACC void eigenValues(Matrix3d *m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret) {
#if TEST_DEBUG
  printf("Matrix(0,0): %f\n", (*m)(0, 0));
  printf("Matrix(1,1): %f\n", (*m)(1, 1));
  printf("Matrix(2,2): %f\n", (*m)(2, 2));
#endif
  SelfAdjointEigenSolver<Matrix3d> es;
  es.computeDirect(*m);
  (*ret) = es.eigenvalues();
  return;
}

struct kernel {
  template< typename T_Acc >
  ALPAKA_FN_ACC
  void operator()(T_Acc const & acc, Matrix3d *m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret) const {
    eigenValues(m, ret);
  }
};

struct kernelInverse3x3 {
  template< typename T_Acc >
    ALPAKA_FN_ACC
    void operator()(T_Acc const & acc, Matrix3d *in, Matrix3d *out) const {
      (*out) = in->inverse();
    }
};

struct kernelInverse4x4 {
  template< typename T_Acc >
    ALPAKA_FN_ACC
    void operator()(T_Acc const & acc, Matrix4d *in, Matrix4d *out) const {
      (*out) = in->inverse();
    }
};


struct kernelInverse5x5 {
  template< typename T_Acc >
    ALPAKA_FN_ACC
    void operator()(T_Acc const & acc, Matrix5d *in, Matrix5d *out) const {
      (*out) = in->inverse();
    }
};


void testInverse3x3() {
  std::cout << "TEST INVERSE 3x3" << std::endl;
  Matrix3d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix3d m_inv = m.inverse();
  Matrix3d *mGPU = nullptr;
  Matrix3d *mGPUret = nullptr;
  Matrix3d *mCPUret = new Matrix3d();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
  std::cout << "Its inverse is:" << std::endl << m.inverse() << std::endl;
#endif
  cudaMalloc((void **)&mGPU, sizeof(Matrix3d));
  cudaMalloc((void **)&mGPUret, sizeof(Matrix3d));
  cudaMemcpy(mGPU, &m, sizeof(Matrix3d), cudaMemcpyHostToDevice);

  CUPLA_KERNEL(kernelInverse3x3)(1, 1, 0, 0)(mGPU, mGPUret);
  cudaDeviceSynchronize();

  cudaMemcpy(mCPUret, mGPUret, sizeof(Matrix3d), cudaMemcpyDeviceToHost);
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << (*mCPUret) << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, *mCPUret));
 }

void testInverse4x4() {
  std::cout << "TEST INVERSE 4x4" << std::endl;
  Matrix4d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix4d m_inv = m.inverse();
  Matrix4d *mGPU = nullptr;
  Matrix4d *mGPUret = nullptr;
  Matrix4d *mCPUret = new Matrix4d();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
  std::cout << "Its inverse is:" << std::endl << m.inverse() << std::endl;
#endif
  cudaMalloc((void **)&mGPU, sizeof(Matrix4d));
  cudaMalloc((void **)&mGPUret, sizeof(Matrix4d));
  cudaMemcpy(mGPU, &m, sizeof(Matrix4d), cudaMemcpyHostToDevice);

  CUPLA_KERNEL(kernelInverse4x4)(1,1,0,0)(mGPU, mGPUret);
  cudaDeviceSynchronize();

  cudaMemcpy(mCPUret, mGPUret, sizeof(Matrix4d), cudaMemcpyDeviceToHost);
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << (*mCPUret) << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, *mCPUret));
}

void testInverse5x5() {
  std::cout << "TEST INVERSE 5x5" << std::endl;
  Matrix5d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix5d m_inv = m.inverse();
  Matrix5d *mGPU = nullptr;
  Matrix5d *mGPUret = nullptr;
  Matrix5d *mCPUret = new Matrix5d();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
  std::cout << "Its inverse is:" << std::endl << m.inverse() << std::endl;
#endif
  cudaMalloc((void **)&mGPU, sizeof(Matrix5d));
  cudaMalloc((void **)&mGPUret, sizeof(Matrix5d));
  cudaMemcpy(mGPU, &m, sizeof(Matrix5d), cudaMemcpyHostToDevice);

  CUPLA_KERNEL(kernelInverse5x5)(1,1,0,0)(mGPU, mGPUret);
  cudaDeviceSynchronize();

  cudaMemcpy(mCPUret, mGPUret, sizeof(Matrix5d), cudaMemcpyDeviceToHost);
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << (*mCPUret) << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, *mCPUret));
}



template <typename M1, typename M2, typename M3>
struct kernelMultiply {
  template< typename T_Acc>
    ALPAKA_FN_ACC
    void operator()(T_Acc const & acc, M1 *J, M2 *C, M3 *result) const {
      //  Map<M3> res(result->data());
#if TEST_DEBUG
      printf("*** GPU IN ***\n");
#endif
//      printIt(J);
//      printIt(C);
      //  res.noalias() = (*J) * (*C);
      //  printIt(&res);
      (*result) = (*J) * (*C);
      printIt(result);
#if TEST_DEBUG
      printf("*** GPU OUT ***\n");
#endif
      return;
    }
};

template <int row1, int col1, int row2, int col2>
void testMultiply() {
  std::cout << "TEST MULTIPLY" << std::endl;
  std::cout << "Product of type " << row1 << "x" << col1 << " * " << row2 << "x" << col2 << std::endl;
  Eigen::Matrix<double, row1, col1> J;
  fillMatrix(J);
  Eigen::Matrix<double, row2, col2> C;
  fillMatrix(C);
  Eigen::Matrix<double, row1, col2> multiply_result = J * C;
#if TEST_DEBUG
  std::cout << "Input J:" << std::endl;
  printIt(&J);
  std::cout << "Input C:" << std::endl;
  printIt(&C);
  std::cout << "Output:" << std::endl;
  printIt(&multiply_result);
#endif
// #ifdef GPU// GPU
  Eigen::Matrix<double, row1, col1> *JGPU = nullptr;
  Eigen::Matrix<double, row2, col2> *CGPU = nullptr;
  Eigen::Matrix<double, row1, col2> *multiply_resultGPU = nullptr;
  Eigen::Matrix<double, row1, col2> *multiply_resultGPUret = new Eigen::Matrix<double, row1, col2>();

  cudaMalloc((void **)&JGPU, sizeof(Eigen::Matrix<double, row1, col1>));
  cudaMalloc((void **)&CGPU, sizeof(Eigen::Matrix<double, row2, col2>));
  cudaMalloc((void **)&multiply_resultGPU, sizeof(Eigen::Matrix<double, row1, col2>));
  cudaMemcpy(JGPU, &J, sizeof(Eigen::Matrix<double, row1, col1>), cudaMemcpyHostToDevice);
  cudaMemcpy(CGPU, &C, sizeof(Eigen::Matrix<double, row2, col2>), cudaMemcpyHostToDevice);
  cudaMemcpy(multiply_resultGPU, &multiply_result, sizeof(Eigen::Matrix<double, row1, col2>), cudaMemcpyHostToDevice);

  CUPLA_KERNEL_OPTI(kernelMultiply<Eigen::Matrix<double, row1, col1>,
      Eigen::Matrix<double, row2, col2>,
      Eigen::Matrix<double, row1, col2> >)(1,1,0,0)(JGPU, CGPU, multiply_resultGPU);
  cudaDeviceSynchronize();

    cudaMemcpy(
      multiply_resultGPUret, multiply_resultGPU, sizeof(Eigen::Matrix<double, row1, col2>), cudaMemcpyDeviceToHost);
  printIt(multiply_resultGPUret);
  assert(isEqualFuzzy(multiply_result, (*multiply_resultGPUret)));
// #endif
}

void testEigenvalues() {
  std::cout << "TEST EIGENVALUES" << std::endl;
  Matrix3d m;
  fillMatrix(m);
  m += m.transpose().eval();

  Matrix3d *m_gpu = nullptr;
  Matrix3d *mgpudebug = new Matrix3d();
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret =
      new Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret1 =
      new Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;
  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret_gpu = nullptr;
  eigenValues(&m, ret);
#if TEST_DEBUG
  std::cout << "Generated Matrix M 3x3:\n" << m << std::endl;
  std::cout << "The eigenvalues of M are:" << std::endl << (*ret) << std::endl;
  std::cout << "*************************\n\n" << std::endl;
#endif
  cudaMalloc((void **)&m_gpu, sizeof(Matrix3d));
  cudaMalloc((void **)&ret_gpu, sizeof(Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType));
  cudaMemcpy(m_gpu, &m, sizeof(Matrix3d), cudaMemcpyHostToDevice);

  CUPLA_KERNEL(kernel)(1,1,0,0)(m_gpu, ret_gpu);
  cudaDeviceSynchronize();

  cudaMemcpy(mgpudebug, m_gpu, sizeof(Matrix3d), cudaMemcpyDeviceToHost);
  cudaMemcpy(ret1, ret_gpu, sizeof(Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType), cudaMemcpyDeviceToHost);
#if TEST_DEBUG
  std::cout << "GPU Generated Matrix M 3x3:\n" << (*mgpudebug) << std::endl;
  std::cout << "GPU The eigenvalues of M are:" << std::endl << (*ret1) << std::endl;
  std::cout << "*************************\n\n" << std::endl;
#endif
  assert(isEqualFuzzy(*ret, *ret1));
}

// template<int row1, int col1, int row2, int col2>
void testEigen()
{

    testMultiply<3, 2, 2, 3>();
    testInverse3x3();
    testInverse4x4();
    testEigenvalues();
    return;
    }

}//end namespace CMS_ARCHITECTURE
