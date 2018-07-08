//#ifdef CUDA
// #include <gpd/net_gpu.h>
//#else
 //#include <gpd/net_cpu.h>
//#endif

#include <gpd/net_gpu.h>

#include <cublas_v2.h>


void Net::gpuBlasMmul(const float *A, const float *B, float *C, const int m, const int k, const int n)
{
  int lda=m,ldb=k,ldc=m;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  cublasHandle_t handle;
  cublasCreate(&handle);

  //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  cublasDestroy(handle);
}


int main() 
{
  Net n;
  return 0;
}