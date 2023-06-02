import os
from openvino.tools import mo
from openvino.runtime import serialize

MODEL_NAME = "yolov5l"

MODEL_PATH = f"../onnx_models/{MODEL_NAME}_openvino_model"
assert MODEL_NAME in ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]
os.makedirs(MODEL_PATH, exist_ok=True)

onnx_path = f"{MODEL_PATH}/{MODEL_NAME}.onnx"

'''
--reverse_input_channels        # to get RGB mode
--scale 255                     # 0-255 to 0.0-1.0
'''
mo_kwargs = {
    "scale": 255,
    "reverse_input_channels": True
}

# fp32 IR model
fp32_path = f"{MODEL_PATH}/FP32/{MODEL_NAME}"
output_path = fp32_path + ".xml"

print(f"Export ONNX to OpenVINO FP32 IR to: {output_path}")
model = mo.convert_model(onnx_path, **mo_kwargs)
serialize(model, output_path)

# fp16 IR model
fp16_path = f"{MODEL_PATH}/FP16/{MODEL_NAME}"
output_path = fp16_path + ".xml"

print(f"Export ONNX to OpenVINO FP16 IR to: {output_path}")
model = mo.convert_model(onnx_path, data_type="FP16", compress_to_fp16=True, 
                            **mo_kwargs)
serialize(model, output_path)