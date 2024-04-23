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

class PytorchSSD(ObjectDetectorMixin, PyTorchEstimator):
    
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


        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This estimator requires un-normalized input images with clip_vales=(0, max_value).")
            if self.clip_values[1] <= 0:  # pragma: no cover
                raise ValueError("This estimator requires un-normalized input images with clip_vales=(0, max_value).")

        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        self._model: torch.nn.Module
        self._model.to(self._device)


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
    @property
    def native_label_is_pytorch_format(self) -> bool:
        """
        Return are the native labels in PyTorch format [x1, y1, x2, y2]?

        :return: Are the native labels in PyTorch format [x1, y1, x2, y2]?
        """
        return True
    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError
    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape
    
    
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

        self._model.train() 

        # Apply preprocessing and convert to tensors
        x_preprocessed, y_preprocessed_list = self._preprocess_and_convert_inputs(x=x, y=y, fit=False, no_grad=False)
        
        # Extract height and width
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]


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


        # Set gradients again after inputs are moved to another device
        if x_preprocessed.is_leaf:
            x_preprocessed.requires_grad = True
        else:
            x_preprocessed.retain_grad()

        # Calculate loss components

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

        

        self._model.eval()
        
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
                # print("\n x_batch.shape : ", x_batch.shape)
                 predictions_x1y1x2y2 = self._model(x_batch)
                

            for prediction_x1y1x2y2 in predictions_x1y1x2y2:
                prediction = {}
                
                prediction["boxes"] = prediction_x1y1x2y2["boxes"].detach().cpu().numpy()
                prediction["labels"] = prediction_x1y1x2y2["labels"].detach().cpu().numpy()
                prediction["scores"] = prediction_x1y1x2y2["scores"].detach().cpu().numpy()
                if "masks" in prediction_x1y1x2y2:
                    prediction["masks"] = prediction_x1y1x2y2["masks"].detach().cpu().numpy().squeeze()
                predictions.append(prediction)           
                             

        return predictions
    
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
        # print("loss_components ", loss_components)
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