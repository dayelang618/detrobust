# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the task specific estimator for PyTorch YOLO v3 and v5 object detectors.

| Paper link: https://arxiv.org/abs/1804.02767
"""
import logging
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.object_detection.utils import cast_inputs_to_pt
from art.estimators.pytorch import PyTorchEstimator

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


def translate_predictions_xcycwh_to_x1y1x2y2(
    y_pred_xcycwh: "torch.Tensor", height: int, width: int
) -> List[Dict[str, "torch.Tensor"]]:
    """
    Convert object detection predictions from xcycwh (YOLO) to x1y1x2y2 (torchvision).

    :param y_pred_xcycwh: Object detection labels in format xcycwh (YOLO).
    :param height: Height of images in pixels.
    :param width: Width if images in pixels.
    :return: Object detection labels in format x1y1x2y2 (torchvision).
    """
    import torch

    y_pred_x1y1x2y2 = []
    device = y_pred_xcycwh.device

    for y_pred in y_pred_xcycwh:
        boxes = torch.vstack(
            [
                torch.maximum((y_pred[:, 0] - y_pred[:, 2] / 2), torch.tensor(0, device=device)),
                torch.maximum((y_pred[:, 1] - y_pred[:, 3] / 2), torch.tensor(0, device=device)),
                torch.minimum((y_pred[:, 0] + y_pred[:, 2] / 2), torch.tensor(height, device=device)),
                torch.minimum((y_pred[:, 1] + y_pred[:, 3] / 2), torch.tensor(width, device=device)),
            ]
        ).permute((1, 0))
        labels = torch.argmax(y_pred[:, 5:], dim=1, keepdim=False)
        scores = y_pred[:, 4]

        y_i = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

        y_pred_x1y1x2y2.append(y_i)

    return y_pred_x1y1x2y2


def translate_labels_x1y1x2y2_to_xcycwh(
    labels_x1y1x2y2: List[Dict[str, "torch.Tensor"]], height: int, width: int
) -> "torch.Tensor":
    """
    Translate object detection labels from x1y1x2y2 (torchvision) to xcycwh (YOLO).

    :param labels_x1y1x2y2: Object detection labels in format x1y1x2y2 (torchvision).
    :param height: Height of images in pixels.
    :param width: Width if images in pixels.
    :return: Object detection labels in format xcycwh (YOLO).
    """
    import torch

    labels_xcycwh_list = []
    device = labels_x1y1x2y2[0]["boxes"].device

    for i, label_dict in enumerate(labels_x1y1x2y2):
        # create 2D tensor to encode labels and bounding boxes
        labels = torch.zeros(len(label_dict["boxes"]), 6, device=device)
        labels[:, 0] = i
        labels[:, 1] = label_dict["labels"]
        labels[:, 2:6] = label_dict["boxes"]

        # normalize bounding boxes to [0, 1]
        labels[:, 2:6:2] = labels[:, 2:6:2] / width
        labels[:, 3:6:2] = labels[:, 3:6:2] / height

        # convert from x1y1x2y2 to xcycwh
        labels[:, 4] -= labels[:, 2]
        labels[:, 5] -= labels[:, 3]
        labels[:, 2] += labels[:, 4] / 2
        labels[:, 3] += labels[:, 5] / 2
        labels_xcycwh_list.append(labels)

    labels_xcycwh = torch.vstack(labels_xcycwh_list)

    return labels_xcycwh


class PyTorchYolo(ObjectDetectorMixin, PyTorchEstimator):
    """
    This module implements the model- and task specific estimator for YOLO v3, v5 object detector models in PyTorch.

    | Paper link: https://arxiv.org/abs/1804.02767
    """

    estimator_params = PyTorchEstimator.estimator_params + ["input_shape", "optimizer", "attack_losses"]

    def __init__(
        self,
        model: "torch.nn.Module",
        input_shape: Tuple[int, ...] = (3, 416, 416),
        optimizer: Optional["torch.optim.Optimizer"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = True,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = (
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type: str = "gpu",
        model_type: str = "ultralytics_yolov8",
    ):
        """
        Initialization.

        :param model: YOLO v3 or v5 model wrapped as demonstrated in examples/get_started_yolo.py.
                      The output of the model is `List[Dict[str, torch.Tensor]]`, one for each input image.
                      The fields of the Dict are as follows:

                      - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                        0 <= y1 < y2 <= H.
                      - labels [N]: the labels for each image.
                      - scores [N]: the scores of each prediction.
        :param input_shape: The shape of one input sample.
        :param optimizer: The optimizer for training the classifier.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_classifier', 'loss_box_reg',
                              'loss_objectness', and 'loss_rpn_box_reg'.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        :param model_type: yolov8 or not //09 Feb 2024 zjw
        """
        import torch

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
        )

        self._input_shape = input_shape
        self._optimizer = optimizer
        self._attack_losses = attack_losses
        self.model_type = model_type

        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This estimator requires un-normalized input images with clip_vales=(0, max_value).")
            if self.clip_values[1] <= 0:  # pragma: no cover
                raise ValueError("This estimator requires un-normalized input images with clip_vales=(0, max_value).")

        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        self._model: torch.nn.Module
        self._model.to(self._device)
       # self._model.eval() 

    @property
    def native_label_is_pytorch_format(self) -> bool:
        """
        Return are the native labels in PyTorch format [x1, y1, x2, y2]?

        :return: Are the native labels in PyTorch format [x1, y1, x2, y2]?
        """
        return True

    @property
    def model(self) -> "torch.nn.Module":
        """
        Return the model.

        :return: The model.
        """
        return self._model

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape

    @property
    def optimizer(self) -> Optional["torch.optim.Optimizer"]:
        """
        Return the optimizer.

        :return: The optimizer.
        """
        return self._optimizer

    @property
    def attack_losses(self) -> Tuple[str, ...]:
        """
        Return the combination of strings of the loss components.

        :return: The combination of strings of the loss components.
        """
        return self._attack_losses

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    def _preprocess_and_convert_inputs(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Optional[List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]] = None,
        fit: bool = False,
        no_grad: bool = True,
    ) -> Tuple["torch.Tensor", List[Dict[str, "torch.Tensor"]]]:
        """
        Apply preprocessing on inputs `(x, y)` and convert to tensors, if needed.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :param fit: `True` if the function is call before fit/training and `False` if the function is called before a
                    predict operation.
        :param no_grad: `True` if no gradients required.
        :return: Preprocessed inputs `(x, y)` as tensors.
        """
        import torch

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0

        if self.all_framework_preprocessing:
            # Convert samples into tensor
            x_tensor, y_tensor = cast_inputs_to_pt(x, y)

            if not self.channels_first:
                x_tensor = torch.permute(x_tensor, (0, 3, 1, 2))
            x_tensor = x_tensor / norm_factor

            # Set gradients
            if not no_grad:
                if x_tensor.is_leaf:
                    x_tensor.requires_grad = True
                else:
                    x_tensor.retain_grad()

            # Apply framework-specific preprocessing
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x=x_tensor, y=y_tensor, fit=fit, no_grad=no_grad)

        elif isinstance(x, np.ndarray):
            # Apply preprocessing
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x=x, y=y, fit=fit, no_grad=no_grad)

            # Convert inputs into tensor
            x_preprocessed, y_preprocessed = cast_inputs_to_pt(x_preprocessed, y_preprocessed)

            if not self.channels_first:
                x_preprocessed = torch.permute(x_preprocessed, (0, 3, 1, 2))
            x_preprocessed = x_preprocessed / norm_factor

            # Set gradients
            if not no_grad:
                x_preprocessed.requires_grad = True

        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        return x_preprocessed, y_preprocessed

    def _get_losses(
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]],
        y_mmdetection=None
    ) -> Tuple[Dict[str, "torch.Tensor"], "torch.Tensor"]:
        """
        Get the loss tensor output of the model including all preprocessing.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Loss components and gradients of the input `x`.
        """
        if self.model_type != 'ultralytics_yolov8':
            self._model.train() 
        # 感觉这里只是想打开模型的训练模式, 但是yolov8就会自己train一次
        else:
            self._model.model.model.train()


        # Apply preprocessing and convert to tensors
        x_preprocessed, y_preprocessed_list = self._preprocess_and_convert_inputs(x=x, y=y, fit=False, no_grad=False)
        
        # Extract height and width
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]

        # Convert labels to YOLO format
        # y_preprocessed_yolo = translate_labels_x1y1x2y2_to_xcycwh(
        #     labels_x1y1x2y2=y_preprocessed, height=height, width=width
        # )

        # Move inputs to device
        x_preprocessed = x_preprocessed.to(self.device)
        import torch
        def move_to_cuda(obj):
            if isinstance(obj, dict):
                return {k: move_to_cuda(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_to_cuda(v) for v in obj]
            elif isinstance(obj, torch.Tensor):
                return obj.cuda()
            else:
                return obj
        y_preprocessed = [move_to_cuda(temp) for temp in y_preprocessed_list]
        
        # y_preprocessed_yolo = y_preprocessed_yolo.to(self.device)

        # Set gradients again after inputs are moved to another device
        if x_preprocessed.is_leaf:
            x_preprocessed.requires_grad = True
        else:
            x_preprocessed.retain_grad()

        # Calculate loss components
        # loss_components = self._model(x_preprocessed, y_preprocessed_yolo, y_mmdetection=y_mmdetection)
        loss_components = self._model(x_preprocessed, y_preprocessed, y_mmdetection=y_mmdetection)


        return loss_components, x_preprocessed

    def loss_gradient(  # pylint: disable=W0613
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], 
        y_mmdetection=None, **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Loss gradients of the same shape as `x`.
        """
        import torch

        loss_components, x_grad = self._get_losses(x=x, y=y, y_mmdetection=y_mmdetection)

        # Compute the gradient and return
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = loss_components[loss_name]
            else:
                loss = loss + loss_components[loss_name]
        print("####### total loss: ", loss, "  #######")
        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward(retain_graph=True)  # type: ignore

        if x_grad.grad is not None:
            if isinstance(x, np.ndarray):
                grads = x_grad.grad.cpu().numpy()
            else:
                grads = x_grad.grad.clone()
        else:
            raise ValueError("Gradient term in PyTorch model is `None`.")

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        if not self.channels_first:
            if isinstance(x, np.ndarray):
                grads = np.transpose(grads, (0, 2, 3, 1))
            else:
                grads = torch.permute(grads, (0, 2, 3, 1))

        assert grads.shape == x.shape

        return grads

    def predict(self, x: np.ndarray, batch_size: int = 128,  **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape NCHW or NHWC.
        :param batch_size: Batch size.
        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict
                 are as follows:

                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                 - labels [N]: the labels for each image.
                 - scores [N]: the scores of each prediction.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Set model to evaluation mode
        # self._model.eval()
        # self._model.training = False # 0215 zjw

        # self._model.model.model.training = False #0220 zjw
        
        if self.model_type != 'ultralytics_yolov8':
            self._model.eval()
        else:
            self._model.model.model.training = False #0221 zjw
        
        # Preprocess inputs and convert to tensors
        x_preprocessed, _ = self._preprocess_and_convert_inputs(x=x, y=None, fit=False, no_grad=True)
    
    
        # Create a DataLoader for batch processing
        dataset = TensorDataset(x_preprocessed)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        # Extract height and width
        # really? NCHW or NHWC ???
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]

        predictions: List[Dict[str, np.ndarray]] = []
        for (x_batch,) in dataloader:
            # Move batch to the specified device
            x_batch = x_batch.to(self._device)
            
            # Compute predictions
            with torch.no_grad():
                print("\n x_batch.shape : ", x_batch.shape)
                predictions_yolo = self._model(x_batch)
                

            # Convert predictions to a different format if the model is not ultralytics_yolov8
            if self.model_type == "ultralytics_yolov8":
                predictions_x1y1x2y2 = predictions_yolo
                
                for prediction_x1y1x2y2 in predictions_x1y1x2y2:
                    prediction = {}

                    prediction["boxes"] = prediction_x1y1x2y2.boxes.xyxy.detach().cpu().numpy()
                    prediction["labels"] = prediction_x1y1x2y2.boxes.cls.detach().cpu().numpy()
                    prediction["scores"] = prediction_x1y1x2y2.boxes.conf.detach().cpu().numpy()
                    if "masks" in prediction_x1y1x2y2:
                        prediction["masks"] = prediction_x1y1x2y2["masks"].detach().cpu().numpy().squeeze()

                predictions.append(prediction)    
            elif self.model_type == "yolov3":
                predictions_x1y1x2y2 = predictions_yolo
                for prediction_x1y1x2y2 in predictions_x1y1x2y2:
                    prediction = {}
                    prediction["boxes"] = prediction_x1y1x2y2["boxes"].detach().cpu().numpy()
                    prediction["labels"] = prediction_x1y1x2y2["labels"].detach().cpu().numpy()
                    prediction["scores"] = prediction_x1y1x2y2["scores"].detach().cpu().numpy()
                    if "masks" in prediction_x1y1x2y2:
                        prediction["masks"] = prediction_x1y1x2y2["masks"].detach().cpu().numpy().squeeze()
                predictions.append(prediction)           

                return predictions                                  
            else:
                predictions_x1y1x2y2 = translate_predictions_xcycwh_to_x1y1x2y2(y_pred_xcycwh=predictions_yolo, height=height, width=width)
                for prediction_x1y1x2y2 in predictions_x1y1x2y2:
                    prediction = {}
                    prediction["boxes"] = prediction_x1y1x2y2["boxes"].detach().cpu().numpy()
                    prediction["labels"] = prediction_x1y1x2y2["labels"].detach().cpu().numpy()
                    prediction["scores"] = prediction_x1y1x2y2["scores"].detach().cpu().numpy()
                    if "masks" in prediction_x1y1x2y2:
                        prediction["masks"] = prediction_x1y1x2y2["masks"].detach().cpu().numpy().squeeze()
                predictions.append(prediction)           

        return predictions

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]],
        batch_size: int = 128,
        nb_epochs: int = 10,
        drop_last: bool = False,
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
                          the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
                          the last batch will be smaller. (default: ``False``)
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
                       and providing it takes no effect.
        """
        import torch
        from torch.utils.data import Dataset, DataLoader

        # Set model to train mode
        self._model.train()

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        # Apply preprocessing and convert to tensors
        x_preprocessed, y_preprocessed = self._preprocess_and_convert_inputs(x=x, y=y, fit=True, no_grad=True)

        class ObjectDetectionDataset(Dataset):
            """
            Object detection dataset in PyTorch.
            """

            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        # Create dataloader
        dataset = ObjectDetectionDataset(x_preprocessed, y_preprocessed)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            collate_fn=lambda batch: list(zip(*batch)),
        )

        # Extract height and width
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]

        # Start training
        for _ in range(nb_epochs):
            # Train for one epoch
            for x_batch, y_batch in dataloader:
                # Convert labels to YOLO
                x_batch = torch.stack(x_batch)
                y_batch = translate_labels_x1y1x2y2_to_xcycwh(labels_x1y1x2y2=y_batch, height=height, width=width)

                # Move inputs to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Form the loss function
                loss_components = self._model(x_batch, y_batch)
                if isinstance(loss_components, dict):
                    loss = sum(loss_components.values())
                else:
                    loss = loss_components

                # Do training
                loss.backward()  # type: ignore
                self._optimizer.step()

            if scheduler is not None:
                scheduler.step()

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_losses(
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute all loss components.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Dictionary of loss components.
        """
        loss_components, _ = self._get_losses(x=x, y=y)
        output = {}
        for key, value in loss_components.items():
            output[key] = value.detach().cpu().numpy()
        return output

    def compute_loss(  # type: ignore
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]],
        y_mmdetection=None, **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape NCHW or NHWC.
        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.
                  The fields of the Dict are as follows:

                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                  - labels [N]: the labels for each image.
        :return: Loss.
        """
        import torch

        loss_components, _ = self._get_losses(x=x, y=y, y_mmdetection=y_mmdetection)

        # Compute the gradient and return
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = loss_components[loss_name]
            else:
                loss = loss + loss_components[loss_name]

        assert loss is not None

        if isinstance(x, torch.Tensor):
            return loss

        return loss.detach().cpu().numpy()
