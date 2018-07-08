#include <gpd/net_gpu.h>


std::vector<float> Net::forward(const std::vector<float>& x)
{
  thrust::device_vector<float> d_x(x);
  thrust::device_vector<float> d_y;
  
  
  
  std::vector<float> y;
  y.resize(d_y.size());
  thrust::copy(d_y.begin(), d_y.end(), y.begin());  
  
  return y;
}


/*void Net::gpuBlasMmul(const float *A, const float *B, float *C, const int m, const int k, const int n)
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
}*/