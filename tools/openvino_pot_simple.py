"""
Model Quantization with POT

use algorithm: DefaultQuantization
acc_checker: None
calibration dataset: optional
"""
import sys
import os
import torch
import cv2
import numpy as np
from addict import Dict
sys.path.append("..")


from openvino.tools.pot.api import DataLoader
from openvino.tools.pot.engines.simplified_engine import SimplifiedEngine
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.utils.logger import init_logger, get_logger


MODEL_NAME = "yolov5l"
MODEL_PATH = f"../onnx_models/{MODEL_NAME}_openvino_model"
assert MODEL_NAME in ["yolov5s", "yolov5m", "yolov5l"]
fp32_path = f"{MODEL_PATH}/FP32/{MODEL_NAME}"
fp16_path = f"{MODEL_PATH}/FP16/{MODEL_NAME}"
IMAGE_SIZE = 640


'''
Create a simple DataLoader

'''
class SimpleDataLoader(DataLoader):
    """Inherit from DataLoader function and implement for YOLOv5."""

    def __init__(self, config):
        if not isinstance(config, dict):
            config = dict(config)
        super().__init__(config)

        self._data_source = config.data_source
        self._imgsz = config.imgsz
        self._batch_size = 1
        self._stride = 32
        self._pad = 0.5
        self._rect = False
        self._workers = 1
        self._data_loader = self._init_dataloader()
        self._data_iter = iter(self._data_loader)

    def __len__(self):
        return len(self._data_loader.dataset)

    def _init_dataloader(self):
        if not os.path.exists(self._data_source):
            raise ValueError("data source not exists: {}".format(self._data_source))
        
        image_list = os.listdir(self._data_source)
        dataset = []
        for image_name in image_list:
            path = os.path.join(self._data_source, image_name)
            im = cv2.imread(path)
            # TODO transform
            dataset.append([im, 0, path, im.shape])
        return DataLoader()

    def __getitem__(self, item):
        try:
            batch_data = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._data_loader)
            batch_data = next(self._data_iter)

        im, target, path, shape = batch_data

        im = im.float()
        im /= 255
        nb, _, height, width = im.shape
        img = im.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        annotation = dict()
        annotation["image_path"] = path
        annotation["target"] = target
        annotation["batch_size"] = nb
        annotation["shape"] = shape
        annotation["width"] = width
        annotation["height"] = height
        annotation["img"] = img

        return (item, annotation), img


def get_config():
    """Set the configuration of the model
    """
    config = dict()

    model_fp32_config = dict(
        {
            "model_name": f"{MODEL_NAME}_fp32",
            "model": fp32_path + ".xml",
            "weights": fp32_path + ".bin",
        }
    )

    model_fp16_config = dict(
        {
            "model_name": f"{MODEL_NAME}_fp16",
            "model": fp16_path + ".xml",
            "weights": fp16_path + ".bin",
        }
    )

    model_int8_config = dict(
        {
            "model_name": f"{MODEL_NAME}_int8",
            "save_path": f"{MODEL_PATH}/INT8_openvino_model/",
        }
    )

    engine_config = dict(
        {
            "type": "simplified"
        }
    )

    dataset_config = Dict(
        {
            "data_source": "/home/linjiaojiao/remote_mount/datasets/BITVehicle/cropped_images",
            "imgsz": IMAGE_SIZE
        }
    )

    algorithms = [
        {
            "name": "DefaultQuantization",  # or AccuracyAwareQuantization
            "params": {
                "target_device": "CPU",
                "preset": "mixed",
                "stat_subset_size": 300,
            },
        }
    ]

    config["model_fp32"] = model_fp32_config
    config["model_fp16"] = model_fp16_config
    config["model_int8"] = model_int8_config
    config["engine"] = engine_config
    config["dataset"] = dataset_config
    config["algorithms"] = algorithms


    return config


if __name__ == "__main__":
    '''
    Run Quantization Pipeline and Accuracy Verification

    The following 9 steps show how to quantize the model using the POT API. The optimized model and collected min-max values will be saved.
    '''

    """ Download dataset and set config
    """
    print("Run the POT. This will take few minutes...")
    config = get_config()
    init_logger(level="INFO")
    logger = get_logger(__name__)

    # Step 1: Load the model.
    model = load_model(config["model_fp16"])
    
    # Step 2: Initialize the data loader.
    data_loader = SimpleDataLoader(config['dataset'])

    # Step 4: Initialize the engine
    engine = SimplifiedEngine(config["engine"], data_loader)

    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(config["algorithms"], engine)

    # Step 6: Execute the pipeline to calculate Min-Max value
    compressed_model = pipeline.run(model)

    # Step 7 (Optional):  Compress model weights to quantized precision
    #                     in order to reduce the size of final .bin file.
    compress_model_weights(compressed_model)

    # Step 8: Save the compressed model to the desired path.
    optimized_save_dir = config["model_int8"]["save_path"]
    save_model(compressed_model, optimized_save_dir, config["model_int8"]["model_name"])

    # Step 9 (Optional): Evaluate the compressed model. Print the results.
    metric_results_i8 = pipeline.evaluate(compressed_model)

    logger.info("Save quantized model in {}".format(optimized_save_dir))
    logger.info("Quantized INT8 model metric_results: {}".format(metric_results_i8))