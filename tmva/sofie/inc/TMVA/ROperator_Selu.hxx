#ifndef SOFIE_ROPERATOR_SELU
#define SOFIE_ROPERATOR_SELU

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE{

template <typename T>
class ROperator_Selu final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_Selu(){}
   ROperator_Selu(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
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
         throw std::runtime_error("TMVA SOFIE Selu Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("TMVA SOFIE Operator Selu called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      for(auto& i: fShape){
         length *= i;
      }
      out << "\t" << "for (int id = 0; id < " << length << " ; id++){\n";
      out << "\t\t" << "tensor_" << fNY << "[id] = 1.0507009873554804934193349852946 * (std::max(float(0.0), tensor_"  << fNX << "[id]) + std::min(0.0, 1.6732632423543772848170429916717 * (std::exp(" << "tensor_" << fNX << "[id]" <<")-1)));\n";
      out << "\t}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}

   // --- Alpaka GPU implementation ---

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      std::stringstream out;
      out << "struct SeluKernel_" << opName << " {\n";
      out << "  template<typename TAcc, typename T>\n";
      out << "  ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* data, T* out, size_t n) const {\n";
      out << "    const auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      out << "    if (idx < n) {\n";
      out << "      constexpr T alpha  = T(1.6732632423543772848170429916717);\n";
      out << "      constexpr T lambda = T(1.0507009873554804934193349852946);\n";
      out << "      T x = data[idx];\n";
      out << "      out[idx] = lambda * (x > T(0) ? x : alpha * (exp(x) - T(1)));\n";
      out << "    }\n";
      out << "  }\n";
      out << "};\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      std::stringstream out;
      out << "SeluKernel_" << opName << " seluKernel_" << opName << ";\n";
      return out.str();
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      std::stringstream out;
      size_t length = 1;
      for (auto & i : fShape) length *= i;
      out << SP << "auto const elementsPerThread_" << fNX << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << fNX << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNX
          << " = {elementsPerGrid_" << fNX << ", elementsPerThread_" << fNX << "};\n";
      out << SP << "auto const workDiv_" << fNX << " = alpaka::getValidWorkDiv(kernelCfg_" << fNX
          << ", devAcc, seluKernel_" << opName
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNX << ", seluKernel_" << opName
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
          << ", static_cast<Idx>(" << length << "));\n";
      return out.str();
   }

};

}//SOFIE


#endif //SOFIE_ROPERATOR_SELU
