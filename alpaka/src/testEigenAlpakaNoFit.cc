#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <alpaka/alpaka.hpp>

using namespace Eigen;

namespace CMS_ARCHITECTURE{

#include "test_common.h"

using Matrix5d = Matrix<double, 5, 5>;

ALPAKA_FN_ACC void eigenValues(Matrix3d* m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType* ret){
    #if TEST_DEBUG
        printf("Matrix(0,0): %f\n", (*m)(0, 0));
        printf("Matrix(1,1): %f\n", (*m)(1, 1));
        printf("Matrix(2,2): %f\n", (*m)(2, 2));
    #endif
        SelfAdjointEigenSolver<Matrix3d> es;
        es.computeDirect(*m);
        (*ret) = es.eigenvalues();
        return;
}// end eigenValues

struct kernel {
    template< typename T_Acc >
    ALPAKA_FN_ACC
    void operator()(T_Acc& acc, Matrix3d *m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType *ret) const{
        eigenValues(m, ret);
    }//end operator
}; //end kernel

struct kernelInverse3x3{
    template< typename T_Acc>
    ALPAKA_FN_ACC
    void operator()(T_Acc const & acc, Matrix3d *in, Matrix3d *out) const {
        (*out) = in->inverse();
    }//end operator
}; //end kernelInverse3x3

struct kernelInverse4x4{
    template< typename T_Acc>
    ALPAKA_FN_ACC
    void operator()(T_Acc const & acc, Matrix4d *in, Matrix4d *out) const {
        (*out) = in->inverse();
    }//end operator
}; //end kernelInverse4x4

struct kernelInverse5x5{
    template< typename T_Acc>
    ALPAKA_FN_ACC
    void operator()(T_Acc const & acc, Matrix5d *in, Matrix5d *out) const {
        (*out) = in->inverse();
    }//end operator
}; //end kernelInverse5x5

template <typename M1, typename M2, typename M3>
struct kernelMultiply {
    template< typename T_Acc>
    ALPAKA_FN_ACC
    void operator()(T_Acc const & acc, M1 *J, M2 *C, M3 *result) const{
#if TEST_DEBUG
      printf("*** GPU IN ***\n");
#endif
//      printIt(J);
//      printIt(C);
      //  res.noalias() = (*J) * (*C);
      //  printIt(&res);
      (*result) = (*J) * (*C);    
#if TEST_DEBUG
    printf("*** GPU OUT ***\n");
#endif
    return;
    }//end operator
};//end kernelMultiply

template <int row1, int col1, int row2, int col2>
void testMultiply() {
    std::cout << "TEST MULTIPLY" << std::endl;
    std::cout << "Product of type" << row1 << "x" << col1 << "*" << row2 << "x" << col2 << std::endl;

     //Select a device
    auto const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    QueueAcc queue(devAcc);

    //Define the work division
    Idx const numElements(1);
    Idx const elementsPerThread(1);
    alpaka::vec::Vec<Dim, Idx> const extent(numElements);

    //Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));        

    using DataJ =  Matrix<double, row1, col1>;
    using DataC =  Matrix<double, row2, col2>;
    using DataR =  Matrix<double, row1, col2>;

    // //Get the host device for allocating memory on the host.
    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    auto const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // //Allocate 3 host memory buffers
    // using BufHostJ = alpaka::mem::buf::Buf<DevHost, DataJ, Dim, Idx>;
    // using BufHostC = alpaka::mem::buf::Buf<DevHost, DataC, Dim, Idx>;
    // using BufHostR = alpaka::mem::buf::Buf<DevHost, DataR, Dim, Idx>;

    auto bufHostJ(alpaka::mem::buf::alloc<DataJ,Idx>(devHost, numElements));
    auto bufHostC(alpaka::mem::buf::alloc<DataC,Idx>(devHost, numElements));
    auto bufHostResult(alpaka::mem::buf::alloc<DataR,Idx>(devHost, numElements));

    auto* const pBufHostJ(alpaka::mem::view::getPtrNative(bufHostJ));
    auto* const pBufHostC(alpaka::mem::view::getPtrNative(bufHostC));
    auto* const pBufHostResult(alpaka::mem::view::getPtrNative(bufHostResult));
    
    // //Fill matrices
    fillMatrix(*pBufHostJ);
    fillMatrix(*pBufHostC);

#if TEST_DEBUG
    printf("Matrix J:\n");
    printIt(pBufHostJ);
    printf("Matrix C:\n");
    printIt(pBufHostC);
    std::cout << "Output:" << std::endl;
    //printIt((*pBufHostJ)*(*pBufHostC));
#endif
       
    // //Allocate 3 buffers on the accelerator
    // using BufAccJ = alpaka::mem::buf::Buf<DevAcc, DataJ, Dim, Idx>;
    // using BufAccC = alpaka::mem::buf::Buf<DevAcc, DataC, Dim, Idx>;
    // using BufAccR = alpaka::mem::buf::Buf<DevAcc, DataR, Dim, Idx>;

    auto bufAccJ(alpaka::mem::buf::alloc<DataJ, Idx>(devAcc, extent));
    auto bufAccC(alpaka::mem::buf::alloc<DataC, Idx>(devAcc, extent));
    auto bufAccR(alpaka::mem::buf::alloc<DataR, Idx>(devAcc, extent));

    // //copy from Host to Acc
    alpaka::mem::view::copy(queue, bufAccJ, bufHostJ, extent);
    alpaka::mem::view::copy(queue, bufAccC, bufHostC, extent);
    alpaka::mem::view::copy(queue, bufAccR, bufHostResult, extent);

    //Instantiate the kernel function object
    kernelMultiply<DataJ, DataC, DataR> kernel;

    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufAccJ),
        alpaka::mem::view::getPtrNative(bufAccC),
        alpaka::mem::view::getPtrNative(bufAccR)));    

        alpaka::queue::enqueue(queue,taskKernel);
        alpaka::wait::wait(queue); //cudaDeviceSynchronize();

        alpaka::mem::view::copy(queue,bufHostResult,bufAccR,extent);
    
        printIt(pBufHostResult);

    assert(isEqualFuzzy((*pBufHostJ)*(*pBufHostC), (*pBufHostResult)));
    
    return;

}//end testMultiply


void testInverse3x3() {
    std::cout << "TEST INVERSE 3x3" << std::endl;

    auto const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    QueueAcc queue(devAcc);
    //Define the work division
    Idx const numElements(1);
    Idx const elementsPerThread(1);
    alpaka::vec::Vec<Dim, Idx> const extent(numElements);

    //Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));        

    using Data =  Matrix3d;
    // //Get the host device for allocating memory on the host.
    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    auto const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    using BufHost = alpaka::mem::buf::Buf<DevHost, Data, Dim, Idx>;

    auto bufHost(alpaka::mem::buf::alloc<Data,Idx>(devHost,numElements));
    auto bufHostI(alpaka::mem::buf::alloc<Data, Idx>(devHost,numElements));

    auto* const pBufHost(alpaka::mem::view::getPtrNative(bufHost));
    auto* const pBufHostI(alpaka::mem::view::getPtrNative(bufHostI));

    fillMatrix(*pBufHost);
    Matrix3d m_inv = pBufHost->inverse();

#if TEST_DEBUG

    std::cout << "Here is the matrix m:" << std::endl << *pBufHost << std::endl;
    std::cout << "Its inverse is:" << std::endl << m_inv << std::endl;

#endif    

    printf("Matrix 3x3: \n");
    printIt(pBufHost);

    //Allocate buffer on the accelerator
    // using BufAcc = alpaka::mem::buf::Buf<DevAcc, Data, Dim, Idx>;
    auto bufAcc(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));
    auto bufAccI(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));

    //Copy from Host to Device
    alpaka::mem::view::copy(queue, bufAcc, bufHost, extent);
    alpaka::mem::view::copy(queue, bufAccI, bufHostI, extent);

    //Kernel istance
    kernelInverse3x3 kernel;

    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufAcc),
        alpaka::mem::view::getPtrNative(bufAccI)));    

    alpaka::queue::enqueue(queue, taskKernel);
    alpaka::wait::wait(queue); //cudaDeviceSynchronize();

    //copy back to the host
    alpaka::mem::view::copy(queue, bufHostI, bufAccI, extent);

    assert(isEqualFuzzy(m_inv, *pBufHostI));

    return; 
} //end testInverse3x3

void testInverse4x4() {
  std::cout << "TEST INVERSE 4x4" << std::endl;

    auto const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    QueueAcc queue(devAcc);
    //Define the work division
    Idx const numElements(1);
    Idx const elementsPerThread(1);
    alpaka::vec::Vec<Dim, Idx> const extent(numElements);

    //Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));        

    using Data =  Matrix4d;

    // //Get the host device for allocating memory on the host.
    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    auto const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // using BufHost = alpaka::mem::buf::Buf<DevHost, Data, Dim, Idx>;

    auto bufHost(alpaka::mem::buf::alloc<Data,Idx>(devHost,numElements));
    auto bufHostI(alpaka::mem::buf::alloc<Data, Idx>(devHost,numElements));

    auto* const pBufHost(alpaka::mem::view::getPtrNative(bufHost));
    auto* const pBufHostI(alpaka::mem::view::getPtrNative(bufHostI));

    fillMatrix(*pBufHost);
    Matrix4d m_inv = pBufHost->inverse();

#if TEST_DEBUG

    std::cout << "Here is the matrix m:" << std::endl << *pBufHost << std::endl;
    std::cout << "Its inverse is:" << std::endl << m_inv << std::endl;

#endif  
    
    printf("Matrix 4x4: \n");
    printIt(pBufHost);

    //Allocate buffer on the accelerator
    // using BufAcc = alpaka::mem::buf::Buf<DevAcc, Data, Dim, Idx>;
    auto bufAcc(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));
    auto  bufAccI(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));

    //Copy from Host to Device
    alpaka::mem::view::copy(queue, bufAcc, bufHost, extent);
    alpaka::mem::view::copy(queue, bufAccI, bufHostI, extent);

    //Kernel istance
    kernelInverse4x4 kernel;

    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufAcc),
        alpaka::mem::view::getPtrNative(bufAccI)));    

    alpaka::queue::enqueue(queue, taskKernel);
    alpaka::wait::wait(queue); //cudaDeviceSynchronize();
    //copy back to the host
    alpaka::mem::view::copy(queue, bufHostI, bufAccI, extent);

    assert(isEqualFuzzy(m_inv, *pBufHostI));

    return; 
} //end testInverse4x4

void testInverse5x5() {
  std::cout << "TEST INVERSE 5x5" << std::endl;

    auto const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    QueueAcc queue(devAcc);
    //Define the work division
    Idx const numElements(1);
    Idx const elementsPerThread(1);
    alpaka::vec::Vec<Dim, Idx> const extent(numElements);

    //Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));        

    using Data =  Matrix5d;

    // //Get the host device for allocating memory on the host.
    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    auto const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    // using BufHost = alpaka::mem::buf::Buf<DevHost, Data, Dim, Idx>;

    auto bufHost(alpaka::mem::buf::alloc<Data,Idx>(devHost,numElements));
    auto bufHostI(alpaka::mem::buf::alloc<Data, Idx>(devHost,numElements));

    auto* const pBufHost(alpaka::mem::view::getPtrNative(bufHost));
    auto* const pBufHostI(alpaka::mem::view::getPtrNative(bufHostI));

    fillMatrix(*pBufHost);
    Matrix5d m_inv = pBufHost->inverse();

#if TEST_DEBUG

    std::cout << "Here is the matrix m:" << std::endl << *pBufHost << std::endl;
    std::cout << "Its inverse is:" << std::endl << m_inv << std::endl;

#endif  
    
    printf("Matrix 5x5: \n");
    printIt(pBufHost);

    //Allocate buffer on the accelerator
    // using BufAcc = alpaka::mem::buf::Buf<DevAcc, Data, Dim, Idx>;
    auto bufAcc(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));
    auto bufAccI(alpaka::mem::buf::alloc<Data, Idx>(devAcc, extent));

    //Copy from Host to Device
    alpaka::mem::view::copy(queue, bufAcc, bufHost, extent);
    alpaka::mem::view::copy(queue, bufAccI, bufHostI, extent);

    //Kernel istance
    kernelInverse5x5 kernel;

    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufAcc),
        alpaka::mem::view::getPtrNative(bufAccI)));    

    alpaka::queue::enqueue(queue, taskKernel);
    alpaka::wait::wait(queue); //cudaDeviceSynchronize();

    //copy back to the host
    alpaka::mem::view::copy(queue, bufHostI, bufAccI, extent);

    assert(isEqualFuzzy(m_inv, *pBufHostI));

    return; 
} //end testInverse5x5


void testEigenvalues() {
    std::cout << "TEST EIGENVALUES" << std::endl;
    
    //get device by index
    auto const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    QueueAcc queue(devAcc);
    //Define work division
    Idx const numElements(1);
    Idx const elementsPerThread(1);
    alpaka::vec::Vec<Dim, Idx> const extent(numElements);

    //Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));  

    using Matrix = Matrix3d;
    using RVector = Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;


    //Get the host device for allocating memory on the host.
    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    auto const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    
    using BufHostMatrix = alpaka::mem::buf::Buf<DevHost, Matrix, Dim, Idx>;
    using BufHostVector = alpaka::mem::buf::Buf<DevHost, RVector, Dim, Idx>;

    auto bufHostMat(alpaka::mem::buf::alloc<Matrix, Idx>(devHost, numElements));
    auto bufHostVector(alpaka::mem::buf::alloc<RVector, Idx>(devHost, numElements));

    Matrix* const pBufHostMat(alpaka::mem::view::getPtrNative(bufHostMat));
    RVector* const pBufHostVec(alpaka::mem::view::getPtrNative(bufHostVector));
    RVector* const pBufHostVec_test(alpaka::mem::view::getPtrNative(bufHostVector));

    fillMatrix(*pBufHostMat);
    // *pBufHostMat += pBufHostMat->transpose().eval();
    eigenValues(pBufHostMat, pBufHostVec_test);

#if TEST_DEBUG
    std::cout << "Generated Matrix M 3x3:\n" << *pBufHostMat << std::endl;
    std::cout << "The eigenvalues of M are:" << std::endl << *pBufHostVec << std::endl;
#endif

    //Allocate buffer on the accelerator
    // using BufAccMatrix = alpaka::mem::buf::Buf<DevAcc, Matrix, Dim, Idx>;
    // using BufAccVector = alpaka::mem::buf::Buf<DevAcc, RVector, Dim, Idx>;

    auto bufAccMat(alpaka::mem::buf::alloc<Matrix, Idx>(devAcc, extent));
    auto bufAccVector(alpaka::mem::buf::alloc<RVector, Idx>(devAcc, extent));

    //copy from host to device
    alpaka::mem::view::copy(queue, bufAccMat, bufHostMat, extent);
    alpaka::mem::view::copy(queue, bufAccVector, bufHostVector, extent);

    kernel kernel;

    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        alpaka::mem::view::getPtrNative(bufAccMat),
        alpaka::mem::view::getPtrNative(bufAccVector)));

    alpaka::queue::enqueue(queue, taskKernel);
    alpaka::wait::wait(queue); //cudaDeviceSynchronize();

    //copy back to the host
    alpaka::mem::view::copy(queue, bufHostVector, bufAccVector, extent);

    assert(isEqualFuzzy(*pBufHostVec, *pBufHostVec_test));

    std::cout << "Generated Matrix M 3x3:\n" << *pBufHostMat << std::endl;
    std::cout << "The eigenvalues of M are:" << std::endl << *pBufHostVec << std::endl;

    return;
}

void testEigen(){

    testMultiply<3, 2, 2, 1>();
    testInverse3x3();
    testInverse4x4();
    testEigenvalues();
    return;
    }
}