#include <gpd/net/openvino_classifier.h>

namespace gpd {
namespace net {

using namespace InferenceEngine;

namespace 
{
  void setDefaultModelPath(Classifier::Device device, std::string& out_model_xml, std::string& out_trained_file)
  {
    if(device == Classifier::Device::eCPU)
    {
      out_model_xml = std::string(MODELS_DIR) + "/fp32/bottles_boxes_cans_5xNeg.xml";
      out_trained_file = std::string(MODELS_DIR) + "/fp32/bottles_boxes_cans_5xNeg.bin";
    }
    else if(device == Classifier::Device::eVPU || device == Classifier::Device::eVPU)
    {
      out_model_xml = std::string(MODELS_DIR) + "/fp16/bottles_boxes_cans_5xNeg.xml";
      out_trained_file = std::string(MODELS_DIR) + "/fp16/bottles_boxes_cans_5xNeg.bin";
    }
    else
    {
      std::cout << "Unknown device id " << static_cast<size_t>(device) << std::endl;
      throw std::exception();
    }
  }
}
std::map<Classifier::Device, const char*> OpenVinoClassifier::device_map_ = {
  {Classifier::Device::eCPU, "CPU"},
  {Classifier::Device::eGPU, "GPU"},
  {Classifier::Device::eVPU, "MYRIAD"},
  {Classifier::Device::eFPGA, "FPGA"}};


OpenVinoClassifier::OpenVinoClassifier(
  Classifier::Device device,
  const size_t batch_size,
  const std::string& model_file,
  const std::string& trained_file)
{
 // --------------------Load IR Generated by ModelOptimizer (.xml and .bin files)--------------------------------------
  InferenceEngine::Core core;

  std::string default_model, default_trained;
  if(model_file.empty() || trained_file.empty())
  {
    setDefaultModelPath(device, default_model, default_trained);
    network_ = core.ReadNetwork(default_model, default_trained);
  }
  else
  {
    network_ = core.ReadNetwork(model_file, trained_file);
  }

  network_.setBatchSize(batch_size);

  // -----------------------------Prepare input blobs-------------------------------------------------------------------
  auto input_info = network_.getInputsInfo().begin()->second;
  auto input_name = network_.getInputsInfo().begin()->first;
  input_info->setPrecision(Precision::FP32);

  // ---------------------------Prepare output blobs--------------------------------------------------------------------
  auto output_info = network_.getOutputsInfo().begin()->second;
  auto output_name = network_.getOutputsInfo().begin()->first;
  output_info->setPrecision(Precision::FP32);

  // -------------------------Loading model to the plugin---------------------------------------------------------------
  std::cout << "network output: "<< output_name << ", input: " <<  input_name << "\n";

  infer_request_ = core.LoadNetwork(network_, device_map_[device], {}).CreateInferRequest();
  input_blob_ = infer_request_.GetBlob(input_name);
  output_blob_ = infer_request_.GetBlob(output_name);
/*
  // -------------------------Preparing the input blob buffer-----------------------------------------------------------
  auto input_data = input_blob_->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
  batch_image_list_.clear();
  const TensorDesc& input_tensor_desc = input_blob_->getTensorDesc();
  for (int i = 0; i < input_tensor_desc.getDims()[2]; ++i) {
    cv::Mat img(input_tensor_desc.getDims()[0], input_tensor_desc.getDims()[1], CV_32FC1, input_data);
    batch_image_list_.push_back(img);
    input_data += input_tensor_desc.getDims()[0] * input_tensor_desc.getDims()[1];
  }
*/
}

std::vector<float> OpenVinoClassifier::classifyImages(
    const std::vector<std::unique_ptr<cv::Mat>> &image_list) {
  std::vector<float> predictions(0);
  InputsDataMap input_info = network_.getInputsInfo();

  for (const auto &item : input_info) {
    Blob::Ptr input = infer_request_.GetBlob(item.first);
    SizeVector dims = input->getTensorDesc().getDims();
    size_t channels = dims[1];
    size_t rows = dims[2];
    size_t cols = dims[3];
    std::cout << "preparing network input: " << item.first << ", channels: " << channels << ", rows: " << rows << ", cols: " << cols << ", dim[0]: " << dims[0] << ", batchSize: " << getBatchSize() << std::endl;
    size_t image_size = rows * cols;
    auto data =
        input->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    int num_iter = (int)ceil(image_list.size() / (double)getBatchSize());

    for (size_t i = 0; i < num_iter; i++) {
      int n = std::min(getBatchSize(),
                       (int)(image_list.size() - i * getBatchSize()));
      for (size_t b = 0; b < n; b++) {
        int image_id = i * getBatchSize() + b;
        for (int r = 0; r < rows; r++) {
          for (int c = 0; c < cols; c++) {
            for (int ch = 0; ch < channels; ch++) {
              int src_idx[3] = {r, c, ch};
              int dst_idx =
                  b * image_size * channels + ch * image_size + r * cols + c;
              data[dst_idx] = image_list[image_id]->at<uchar>(src_idx);
            }
          }
        }
        if (n < getBatchSize()) {
          for (int b = n; b < getBatchSize(); b++) {
            for (int r = 0; r < rows; r++) {
              for (int c = 0; c < cols; c++) {
                for (int ch = 0; ch < channels; ch++) {
                  int dst_idx = b * image_size * channels + ch * image_size +
                                r * cols + c;
                  data[dst_idx] = 0;
                }
              }
            }
          }
        }
      }
      infer_request_.Infer();

      auto output_data =
          output_blob_->buffer()
              .as<PrecisionTrait<Precision::FP32>::value_type *>();
      //const int resultsCnt = output_blob_->size() / getBatchSize();

      for (int j = 0; j < n; j++) {
        std::cout << "positive score: " << output_data[2 * j + 1] << ", negative score: " << output_data[2 * j] << "\n";
        predictions.push_back(output_data[2 * j + 1] - output_data[2 * j]);
      }
    }
  }

  return predictions;
}

int OpenVinoClassifier::getBatchSize() const { return network_.getBatchSize(); }

}  // namespace net
}  // namespace gpd
