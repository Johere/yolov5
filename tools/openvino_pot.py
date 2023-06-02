"""
https://docs.openvino.ai/latest/notebooks/220-yolov5-accuracy-check-and-quantization-with-output.html
"""
import sys
import os
import torch
import numpy as np
from addict import Dict
sys.path.append("..")

from utils.dataloaders import create_dataloader
from utils.general import (
    check_dataset,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    check_yaml,
)
from utils.metrics import ap_per_class
from val import process_batch

from openvino.tools.pot.api import Metric, DataLoader
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.utils.logger import init_logger, get_logger


MODEL_NAME = "yolov5s"
MODEL_PATH = f"../onnx_models/{MODEL_NAME}_openvino_model"
assert MODEL_NAME in ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]
fp32_path = f"{MODEL_PATH}/FP32/{MODEL_NAME}"
fp16_path = f"{MODEL_PATH}/FP16/{MODEL_NAME}"
IMAGE_SIZE = 640

'''
Create YOLOv5 DataLoader class

Create a class for the loading YOLOv5 dataset and annotation which inherits from POT API class DataLoader. 
The Ultralytics YOLOv5 training process requires image data normalization from [0,225] 8-bit integer range to [0.0,1.0] 32-bit floating point range.

'''
class YOLOv5DataLoader(DataLoader):
    """Inherit from DataLoader function and implement for YOLOv5."""

    def __init__(self, config):
        if not isinstance(config, dict):
            config = dict(config)
        super().__init__(config)

        self._data_source = config.data_source
        self._imgsz = config.imgsz
        self._batch_size = 1
        self._stride = 32
        self._single_cls = config.single_cls
        self._pad = 0.5
        self._rect = False
        self._workers = 1
        self._data_loader = self._init_dataloader()
        self._data_iter = iter(self._data_loader)

    def __len__(self):
        return len(self._data_loader.dataset)

    def _init_dataloader(self):
        dataloader = create_dataloader(
            self._data_source["val"],
            imgsz=self._imgsz,
            to_rgb=False,   # image should be BGR mode, because we let openvino.mo do the `reverse_input_channels``
            batch_size=self._batch_size,
            stride=self._stride,
            single_cls=self._single_cls,
            pad=self._pad,
            rect=self._rect,
            workers=self._workers,
        )[0]
        return dataloader

    def __getitem__(self, item):
        try:
            batch_data = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._data_loader)
            batch_data = next(self._data_iter)

        im, target, path, shape = batch_data

        im = im.float()
        
        # scale should be processed in openvino.mo step
        # im /= 255
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


'''
Create YOLOv5 Metric Class

Create a class to measure the model performance by Mean Average Precision (mAP) 
with the COCO dataset predicted result and annotation value, after applying 
Ultralytics NMS routine (yolov5.utils.general.non_max_suppression). 
Here we use both AP@0.5 and AP@0.5:0.95 as the measurement standard. 
This class should be inherited from the POT API Metric class.
'''
class COCOMetric(Metric):
    """Inherit from DataLoader function and implement for YOLOv5."""

    def __init__(self, config):
        super().__init__()
        self._metric_dict = {"AP@0.5": [], "AP@0.5:0.95": []}
        self._names = (*self._metric_dict,)
        self._stats = []
        self._last_stats = []
        self._conf_thres = config.conf_thres
        self._iou_thres = config.iou_thres
        self._single_cls = config.single_cls
        self._nc = config.nc
        self._class_names = {idx: name for idx, name in enumerate(config.names)}
        self._device = config.device

    @property
    def value(self):
        """Returns metric value for the last model output.
        Both use AP@0.5 and AP@0.5:0.95
        """
        mp, mr, map50, map = self._process_stats(self._last_stats)

        return {self._names[0]: [map50], self._names[1]: [map]}

    @property
    def avg_value(self):
        """Returns metric value for all model outputs.
        Both use AP@0.5 and AP@0.5:0.95
        """
        mp, mr, map50, map = self._process_stats(self._stats)

        return {self._names[0]: map50, self._names[1]: map}

    def _process_stats(self, stats):
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(
                *stats, plot=False, save_dir=None, names=self._class_names
            )
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            np.bincount(stats[3].astype(np.int64), minlength=self._nc)
        else:
            torch.zeros(1)

        return mp, mr, map50, map

    def update(self, output, target):
        """Calculates and updates metric value
        Contains postprocessing part from Ultralytics YOLOv5 project
        :param output: model output
        :param target: annotations
        """

        annotation = target[0]["target"]
        width = target[0]["width"]
        height = target[0]["height"]
        shapes = target[0]["shape"]
        paths = target[0]["image_path"]
        im = target[0]["img"]

        iouv = torch.linspace(0.5, 0.95, 10).to(
            self._device
        )  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        seen = 0
        stats = []
        # NMS
        annotation = torch.Tensor(annotation)
        annotation[:, 2:] *= torch.Tensor([width, height, width, height]).to(
            self._device
        )  # to pixels
        lb = []

        out = np.concatenate(output, axis=1)
        # out = output[0]
        out = torch.Tensor(out).to(self._device)
        out = non_max_suppression(
            out,
            self._conf_thres,
            self._iou_thres,
            labels=lb,
            multi_label=True,
            agnostic=self._single_cls,
        )
        # Metrics
        for si, pred in enumerate(out):
            labels = annotation[annotation[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            shape = shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append(
                        (
                            torch.zeros(0, niou, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls,
                        )
                    )
                continue

            # Predictions
            if self._single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(
                im[si].shape[1:], predn[:, :4], shape, shapes[si][1]
            )  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(
                    im[si].shape[1:], tbox, shape, shapes[si][1]
                )  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            self._stats.append(
                (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
            )
        self._last_stats = stats

    def reset(self):
        """Resets metric"""
        self._metric_dict = {"AP@0.5": [], "AP@0.5:0.95": []}
        self._last_stats = []
        self._stats = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                                                 'type': metric type
        """
        return {
            self._names[0]: {"direction": "higher-better", "type": "AP@0.5"},
            self._names[1]: {"direction": "higher-better", "type": "AP@0.5:0.95"},
        }


'''
Set POT Configuration

Create a function to set the configuration of the model, engine, dataset, metric and algorithms, 
which are used by the POT quantization pipeline API.

'''
def get_config():
    """Set the configuration of the model, engine,
    dataset, metric and quantization algorithm.
    """
    config = dict()
    data_yaml = check_yaml("../data/coco128.yaml")
    # data_yaml = check_yaml("../data/coco-single.yaml")
    data = check_dataset(data_yaml)

    model_fp32_config = Dict(
        {
            "model_name": f"{MODEL_NAME}_fp32",
            "model": fp32_path + ".xml",
            "weights": fp32_path + ".bin",
        }
    )

    model_fp16_config = Dict(
        {
            "model_name": f"{MODEL_NAME}_fp16",
            "model": fp16_path + ".xml",
            "weights": fp16_path + ".bin",
        }
    )

    model_int8_config = Dict(
        {
            "model_name": f"{MODEL_NAME}_int8",
            "save_path": f"{MODEL_PATH}/FP16-INT8/",
        }
    )

    engine_config = Dict(
        {"device": "CPU", "stat_requests_number": 8, "eval_requests_number": 8}
    )

    dataset_config = Dict(
        {
            "data_source": data,
            "imgsz": IMAGE_SIZE,
            "single_cls": False,
        }
    )

    metric_config = Dict(
        {
            "conf_thres": 0.001,
            "iou_thres": 0.65,
            "single_cls": False,
            "nc": 1,  # if opt.single_cls else int(data['nc']),
            "names": data["names"],
            "device": "cpu",
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
    config["metric"] = metric_config
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
    data_loader = YOLOv5DataLoader(config["dataset"])

    # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
    metric = COCOMetric(config["metric"])

    # Step 4: Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(config=config["engine"], data_loader=data_loader, metric=metric)

    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(config["algorithms"], engine)

    metric_results = None

    # Check the FP32 model accuracy.
    metric_results_fp32 = pipeline.evaluate(model)

    logger.info("FP32 model metric_results: {}".format(metric_results_fp32))

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