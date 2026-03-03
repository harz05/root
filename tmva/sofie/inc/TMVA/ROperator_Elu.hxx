#ifndef SOFIE_ROPERATOR_Elu
#define SOFIE_ROPERATOR_Elu

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE{

template <typename T>
class ROperator_Elu final : public ROperator
{

private:

   /* Attributes*/
   float falpha= 1.0; //default value
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::string fType;

public:
   ROperator_Elu(){}
   ROperator_Elu(float alpha,std::string nameX, std::string nameY):
   falpha(alpha),fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = { fNX };
      fOutputTensorNames = { fNY };
      
      if(std::is_same<T, float>::value){
         fType = "float";
      }
		else{
			throw std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a Elu operator");
		}
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Elu Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Elu called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      out << SP << "float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha << ";\n";

      out << "\n//------ ELU \n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] >= 0 )? tensor_" << fNX << "[id] : "<< OpName << "_alpha * std::exp(tensor_"<< fNX<<"[id]) - 1);\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      std::string op;
      op =  "\n//------ ELU_KERNEL_ALPAKA\n";
      op += "struct EluKernel {\n";
      op += SP + "template<typename TAcc, typename T>\n";
      op += SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements, T alpha) const {\n";
      op += SP + SP + "const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + "if (idx < numElements) {\n";
      op += SP + SP + SP + "out[idx] = (data[idx] >= static_cast<T>(0)) ? data[idx] : alpha * (exp(data[idx]) - static_cast<T>(1));\n";
      op += SP + SP + "}\n";
      op += SP + "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "EluKernel eluKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Elu operator called to Generate_GPU_ALPAKA without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertShapeToLength(fShape);
      // alpha is baked into the generated code at model-parse time
      std::string alphaStr = std::to_string(falpha) + "f";
      out << "\n//------ ELU_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNX << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_" << fNX << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNX << " = {elementsPerGrid_" << fNX << ", elementsPerThread_" << fNX << "};\n";
      out << SP << "auto const workDiv_" << fNX << " = alpaka::getValidWorkDiv(kernelCfg_" << fNX << ", devAcc, eluKernel, "
          << "alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_" << fNY
          << "), static_cast<Idx>(" << length << "), static_cast<" << fType << ">(" << alphaStr << "));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNX << ", eluKernel, "
          << "alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_" << fNY
          << "), static_cast<Idx>(" << length << "), static_cast<" << fType << ">(" << alphaStr << "));\n";
      return out.str();
   }

};

}//SOFIE



#endif //SOFIE_ROPERATOR_Elu
