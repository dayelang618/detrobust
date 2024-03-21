# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the task specific estimator for Faster R-CNN v3 in PyTorch.
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
import cv2 
import matplotlib.pyplot as plt 

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    import torchvision

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def extract_predictions(predictions_):
    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    print("\npredicted classes:", predictions_class)

    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = 0.5
    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

    predictions_boxes = predictions_boxes[: predictions_t + 1]
    predictions_class = predictions_class[: predictions_t + 1]

    return predictions_class, predictions_boxes, predictions_class


def plot_image_with_boxes(img, boxes, pred_cls):
    text_size = 5
    text_th = 5
    rect_th = 6
    
    for i in range(len(boxes)):
        # Draw Rectangle with the coordinates

        cv2.rectangle(
            img,
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            (int(boxes[i][1][0]), int(boxes[i][1][1])),
            color=(0, 255, 0),
            thickness=rect_th,
        )
        # Write the prediction class
        cv2.putText(
            img,
            pred_cls[i],
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 255, 0),
            thickness=text_th,
        )
    plt.axis("off")
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    plt.show()
    
    
class PyTorchFasterRCNN(PyTorchObjectDetector):
    """
    This class implements a model-specific object detector using Faster R-CNN and PyTorch following the input and output
    formats of torchvision.
    """

    def __init__(
        self,
        model: Optional["torchvision.models.detection.FasterRCNN"] = None,
        input_shape: Tuple[int, ...] = (-1, -1, -1),
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
        attack_method = "PGD",
    ):
        """
        Initialization.

        :param model: Faster R-CNN model. The output of the model is `List[Dict[str, torch.Tensor]]`, one for
                      each input image. The fields of the Dict are as follows:

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
        :param attack_method: PGD, CW for now
        """
        import torchvision

        if model is None:  # pragma: no cover
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
            )

        super().__init__(
            model=model,
            input_shape=input_shape,
            optimizer=optimizer,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            attack_losses=attack_losses,
            device_type=device_type,
        )
        self.attack_method=attack_method
    
    
    def class_gradient(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        label: Optional[Union[int, List[int], np.ndarray]] = None,
        training_mode: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
                              Note on RNN-like models: Backpropagation through RNN modules in eval mode raises
                              RuntimeError due to cudnn issues and require training mode, i.e. RuntimeError: cudnn RNN
                              backward can only be called in training mode. Therefore, if the model is an RNN type we
                              always use training mode but freeze batch-norm and dropout layers if
                              `training_mode=False.`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        import torch

        self._model.train(mode=training_mode)
        self.nb_classes = 91 # just for detection coco dataset 3.11 zjw

        if isinstance(label, list):
            label = np.array(label)
        if not (
            (label is None)
            or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
            or (
                isinstance(label, np.ndarray)
                and len(label.shape) == 1
                and (label < self.nb_classes).all()
                and label.shape[0] == x.shape[0]
            )
        ):
            raise ValueError(f"Label {label} is out of range.")  # pragma: no cover

        self._layer_idx_gradients = -1
        # Apply preprocessing
        if self.all_framework_preprocessing:
            x_grad = torch.from_numpy(x).to(self._device)
            if self._layer_idx_gradients < 0:
                x_grad.requires_grad = True
            x_input, _ = self._apply_preprocessing(x_grad, y=None, fit=False, no_grad=False)
        else:
            x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False, no_grad=True)
            x_grad = torch.from_numpy(x_preprocessed).to(self._device)
            if self._layer_idx_gradients < 0:
                x_grad.requires_grad = True
            x_input = x_grad

        # Run prediction
        # model_outputs: List[] scores shape: [1,num of classes]
        model_outputs = self._model(x_input)

        model_outputs_class_logits = []
        for output in model_outputs:
            model_outputs_class_logits.append(output["class_logits"])
            
        model_outputs = model_outputs_class_logits
        # Set where to get gradient
        if self._layer_idx_gradients >= 0:
            input_grad = model_outputs[self._layer_idx_gradients]
        else:
            input_grad = x_grad

        # Set where to get gradient from
        # preds = model_outputs[-1]

        # Compute the gradient
        grads_list = []

        def save_grad():
            def hook(grad):
                grads_list.append(grad.cpu().numpy().copy())
                grad.data.zero_()

            return hook

        input_grad.register_hook(save_grad())

        self._model.zero_grad()
        grads_all_list = []
        for i in range(len(model_outputs_class_logits)):
            preds = model_outputs_class_logits[i] # for detection 
            
            if label is None:
                if len(preds.shape) == 1 or preds.shape[1] == 1:
                    num_outputs = 1
                else:
                    num_outputs = self.nb_classes

                for i in range(num_outputs):
                    torch.autograd.backward(
                        preds[:, i],
                        torch.tensor([1.0] * len(preds[:, 0])).to(self._device),
                        retain_graph=True,
                    )

                grads = np.swapaxes(np.array(grads_list), 0, 1)

            elif isinstance(label, (int, np.integer)):
                torch.autograd.backward(
                    preds[:, label],
                    torch.tensor([1.0] * len(preds[:, 0])).to(self._device),
                    retain_graph=True,
                )
                grads = np.swapaxes(np.array(grads_list), 0, 1)
                
                
            else:
                unique_label = list(np.unique(label))
                for i in unique_label:
                    torch.autograd.backward(
                        preds[:, i],
                        torch.tensor([1.0] * len(preds[:, 0])).to(self._device),
                        retain_graph=True,
                    )

                grads = np.swapaxes(np.array(grads_list), 0, 1)
                lst = [unique_label.index(i) for i in label]
                grads = grads[np.arange(len(grads)), lst]

                grads = grads[None, ...]
                grads = np.swapaxes(np.array(grads), 0, 1)

            if not self.all_framework_preprocessing:
                grads = self._apply_preprocessing_gradient(x, grads)
                
            grads_all_list.append(grads)

        return sum(grads_all_list)
    
    
    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> List[Dict[str, np.ndarray]]:
            """
            Perform prediction for a batch of inputs.

            :param x: Samples of shape NCHW.
            :param batch_size: Batch size.
            :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict
                    are as follows:

                    - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                    - labels [N]: the labels for each image
                    - scores [N]: the scores or each prediction.
            """
            import torch
            from torch.utils.data import TensorDataset, DataLoader

            # Set model to evaluation mode
            self._model.eval()

            # Apply preprocessing and convert to tensors
            x_preprocessed, _ = self._preprocess_and_convert_inputs(x=x, y=None, fit=False, no_grad=True)

            # Create dataloader
            dataset = TensorDataset(x_preprocessed)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

            predictions: List[Dict[str, np.ndarray]] = []
            for (x_batch,) in dataloader:
                # Move inputs to device
                x_batch = x_batch.to(self._device)

                # Run prediction
                with torch.no_grad():
                    predictions_x1y1x2y2 = self._model(x_batch)


                for prediction_x1y1x2y2 in predictions_x1y1x2y2:
                    prediction = {}

                    prediction["boxes"] = prediction_x1y1x2y2["boxes"].detach().cpu().numpy()
                    prediction["labels"] = prediction_x1y1x2y2["labels"].detach().cpu().numpy()
                    prediction["scores"] = prediction_x1y1x2y2["scores"].detach().cpu().numpy()
                    if self.attack_method == "CW":
                        prediction["class_logits"] = prediction_x1y1x2y2["class_logits"].cpu().numpy()
                    if "masks" in prediction_x1y1x2y2:
                        prediction["masks"] = prediction_x1y1x2y2["masks"].detach().cpu().numpy().squeeze()

                    predictions.append(prediction)
                    
            return predictions