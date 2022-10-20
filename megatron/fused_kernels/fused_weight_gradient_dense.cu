#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifdef __HIP_PLATFORM_HCC__

inline rocblas_status myGemmEx(
  rocblas_handle handle, 
  rocblas_operation transA, 
  rocblas_operation transB, 
  rocblas_int m, 
  rocblas_int n, 
  rocblas_int k, 
  const void *alpha, 
  const void *a, 
  rocblas_datatype a_type, 
  rocblas_int lda, 
  const void *b, 
  rocblas_datatype b_type, 
  rocblas_int ldb,
  const void *beta, 
  void *c, 
  rocblas_datatype c_type, 
  rocblas_int ldc,
  rocblas_datatype compute_type, 
  rocblas_gemm_algo algo) {
    return  rocblas_gemm_ex(
      handle, 
      transA, 
      transB, 
      m, 
      n, 
      k, 
      alpha, 
      a, 
      a_type, 
      lda, 
      b, 
      b_type, 
      ldb, 
      beta,
      c, 
      c_type, 
      ldc, 
      c, 
      c_type, 
      ldc,
      compute_type, 
      algo, 
      0, 
      0);
  }
# define myAlgo rocblas_gemm_algo_standard
# define mybf16 rocblas_datatype_bf16_r
# define myf16 rocblas_datatype_f16_r
# define myf32 rocblas_datatype_f32_r
#else
# define myGemmEx cublasGemmEx
# define myAlgo CUBLAS_GEMM_DEFAULT_TENSOR_OP
# define mybf16 CUDA_R_16BF
# define myf16 CUDA_R_16F
# define myf32 CUDA_R_32F
#endif


// BF16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemmex_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::BFloat16* A,
    int lda,
    at::BFloat16* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
  return myGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      mybf16,
      lda,
      B,
      mybf16,
      ldb,
      beta,
      C,
      myf32,
      ldc,
      myf32,
      myAlgo);  
}

// FP16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemmex_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
  return myGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      myf16,
      lda,
      B,
      myf16,
      ldb,
      beta,
      C,
      myf32,
      ldc,
      myf32,
      myAlgo);
}

// FP32 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemmex_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    float* A,
    int lda,
    float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
  return myGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      myf32,
      lda,
      B,
      myf32,
      ldb,
      beta,
      C,
      myf32,
      ldc,
      myf32,
      myAlgo);
}

template <typename T>
int wgrad_gemm_accum_fp32_cuda(T *input, T *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha = 1.0;
    const float beta  = 1.0;
    int status = 1;

    status = static_cast<int>(gemmex_wrapper(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        in_dim,
        out_dim,
        hidden_dim,
        &alpha,
        input,
        in_dim,
        d_output,
        out_dim,
        &beta,
        d_weight,
        in_dim));
    return status;
}

template int wgrad_gemm_accum_fp32_cuda<at::Half>(at::Half *input, at::Half *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);
template int wgrad_gemm_accum_fp32_cuda<at::BFloat16>(at::BFloat16 *input, at::BFloat16 *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);
template int wgrad_gemm_accum_fp32_cuda<float>(float *input, float *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);
