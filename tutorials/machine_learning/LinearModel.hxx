//Code generated automatically by TMVA for Inference of Model file [LinearModel.onnx] at [Sun Mar  8 13:33:06 2026] 

#ifndef ROOT_TMVA_SOFIE_LINEARMODEL
#define ROOT_TMVA_SOFIE_LINEARMODEL

#include <algorithm>
#include <cmath>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_LinearModel{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_4weight = std::vector<float>(16);
float * tensor_4weight = fTensor_4weight.data();
std::vector<float> fTensor_2bias = std::vector<float>(8);
float * tensor_2bias = fTensor_2bias.data();
std::vector<float> fTensor_4bias = std::vector<float>(2);
float * tensor_4bias = fTensor_4bias.data();
std::vector<float> fTensor_2weight = std::vector<float>(128);
float * tensor_2weight = fTensor_2weight.data();
std::vector<float> fTensor_0bias = std::vector<float>(16);
float * tensor_0bias = fTensor_0bias.data();
std::vector<float> fTensor_0weight = std::vector<float>(512);
float * tensor_0weight = fTensor_0weight.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(96);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor relu with size 64 bytes
float* tensor_relu = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor relu_1 with size 32 bytes
float* tensor_relu_1 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 64);

 // Allocating memory for intermediate tensor linear_2 with size 8 bytes
float* tensor_linear_2 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 56);

 // Allocating memory for intermediate tensor output with size 8 bytes
float* tensor_output = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 48);


Session(std::string filename ="LinearModel.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_4weight, "tensor_4weight", 16);
   ReadTensorFromStream(f, tensor_2bias, "tensor_2bias", 8);
   ReadTensorFromStream(f, tensor_4bias, "tensor_4bias", 2);
   ReadTensorFromStream(f, tensor_2weight, "tensor_2weight", 128);
   ReadTensorFromStream(f, tensor_0bias, "tensor_0bias", 16);
   ReadTensorFromStream(f, tensor_0weight, "tensor_0weight", 512);
   f.close();

}

void doInfer(float const* tensor_input,  std::vector<float> &output_tensor_output ){


//--------- Gemm op_0 { 1 , 32 } * { 16 , 32 } -> { 1 , 16 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 16 * j;
      for (size_t k = 0; k < 16; k++) { 
         tensor_relu[y_index + k] = tensor_0bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_relu, true, false, 16, 1, 32, 1, tensor_0weight, tensor_input, 1,nullptr);
   for (int id = 0; id < 16 ; id++){
      tensor_relu[id] = ((tensor_relu[id] > 0 )? tensor_relu[id] : 0);
   }

//--------- Gemm op_1 { 1 , 16 } * { 8 , 16 } -> { 1 , 8 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 8 * j;
      for (size_t k = 0; k < 8; k++) { 
         tensor_relu_1[y_index + k] = tensor_2bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_relu_1, true, false, 8, 1, 16, 1, tensor_2weight, tensor_relu, 1,nullptr);
   for (int id = 0; id < 8 ; id++){
      tensor_relu_1[id] = ((tensor_relu_1[id] > 0 )? tensor_relu_1[id] : 0);
   }

//--------- Gemm op_2 { 1 , 8 } * { 2 , 8 } -> { 1 , 2 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 2 * j;
      for (size_t k = 0; k < 2; k++) { 
         tensor_linear_2[y_index + k] = tensor_4bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_linear_2, true, false, 2, 1, 8, 1, tensor_4weight, tensor_relu_1, 1,nullptr);

   //------ SOFTMAX - 2  2  1
   for (int i = 0; i < 1; ++i) {
      size_t offset = i * 2;
      float const * x_ptr = &tensor_linear_2[offset];
      float * y_ptr = &tensor_output[offset];
      float vmax = x_ptr[0];
      for (int j = 1; j < 2; ++j) {
         if (x_ptr[j] > vmax) vmax = x_ptr[j];
      }
      float sum = 0.0;
      for (int j = 0; j < 2; ++j) {
         y_ptr[j] = std::exp(x_ptr[j] - vmax);
         sum += y_ptr[j];
      }
      float inv_sum = 1.0f / sum;
      for (int j = 0; j < 2; ++j) {
         y_ptr[j] *= inv_sum;
      }
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_output, output_tensor_output, 2);
}



std::vector<float> infer(float const* tensor_input){
   std::vector<float > output_tensor_output;
   doInfer(tensor_input, output_tensor_output );
   return {output_tensor_output};
}
};   // end of Session

} //TMVA_SOFIE_LinearModel

#endif  // ROOT_TMVA_SOFIE_LINEARMODEL
