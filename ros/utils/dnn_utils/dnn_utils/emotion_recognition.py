import os
import numpy as np
import torch
from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel
import onnxruntime as ort
from torch2trt import TRTModule

LABELS = ['angry', 'disgust','fearful','happy','neutral','sad','surprised']

class EmotionVideoRecognition(DnnModel):
    def __init__(self, inference_type=None):
        self.interference_type = inference_type
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'emotion_video_1027.onnx')
        self._model = self.load_model(torch_script_model_path, inference_type)

    def __call__(self, input): 
        inputs = {
                    'input': input.numpy()
                 }   
        return self._model.run(None, inputs)[0], LABELS
   
    def load_model(self, torch_script_model_path, inference_type):
        if inference_type == 'torchscript':
            model = ort.InferenceSession(
                torch_script_model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            return model
        else:
            raise ValueError("Unsupported inference type")
    
class EmotionAudioRecognition(DnnModel):
    def __init__(self, inference_type=None):
        self.inference_type = inference_type
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'best_distilhubert_emotion_unfrozen_attnpool.onnx')
        self._model = self.load_model(torch_script_model_path, inference_type)
    
    def __call__(self, input): 
        if isinstance(input, torch.Tensor):
            input = input.detach().cpu().numpy().astype(np.float32)
        else:
            input = input.astype(np.float32)

        # Ensure shape (1, T)
        if input.ndim == 1:
            input = input[None, :]
        elif input.ndim != 2:
            raise ValueError(f"Unexpected input shape {input.shape}, expected (T,) or (1, T)")

        ort_inputs = {"input": input}
        logits = self._model.run(None, ort_inputs)[0]
        return logits, LABELS

    def load_model(self, torch_script_model_path, inference_type):
        if inference_type == 'torchscript':
            if not os.path.exists(torch_script_model_path):
                raise FileNotFoundError(f"ONNX model not found at: {torch_script_model_path}")

            # Choose valid providers
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "CPUExecutionProvider" in available:
                providers = ["CPUExecutionProvider"]
            else:
                raise RuntimeError(f"No valid ONNX Runtime providers: {available}")

            so = ort.SessionOptions()
            model = ort.InferenceSession(
                torch_script_model_path,
                sess_options=so,
                providers=providers,
            )
            return model
        else:
            raise ValueError("Unsupported inference type")

class EmotionTextRecognition(DnnModel):
    def __init__(self, inference_type=None):
        self.inference_type = inference_type
        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', 'emotion_text.onnx')
        self._model = self.load_model(torch_script_model_path, inference_type)
        
    def __call__(self, input_ids, attention_mask): 
        inputs = {
                    'input_ids': input_ids.numpy(),
                    'attention_mask': attention_mask.numpy()
                 }   
        return self._model.run(None, inputs)[0], LABELS
   
    def load_model(self, torch_script_model_path, inference_type):
        if inference_type == 'torchscript':
            model = ort.InferenceSession(
                torch_script_model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            return model
        else:
            raise ValueError("Unsupported inference type")
