"""
The script demonstrates a simple example of using ART with YOLO (versions 3 and 5).
The example loads a YOLO model pretrained on the COCO dataset
and creates an adversarial example using Projected Gradient Descent method.

- To use Yolov3, run:
        pip install pytorchyolo

- To use Yolov5, run:
        pip install yolov5

- To use Yolov8 run:
        pip install ultralytics

Note: If pytorchyolo throws an error in pytorchyolo/utils/loss.py, add before line 174 in that file, the following:
        gain = gain.to(torch.int64)
        
Note: update with yolov8 example 07 Feb 2024 by zjw
"""
import sys
sys.path.append("/home/jiawei/data/zjw/ultralytics")
sys.path.append("/home/jiawei/data/zjw/utils")
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from utils import yolov3_loss

from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion.fast_gradient import FastGradientMethod
import cv2
import matplotlib
import matplotlib.pyplot as plt

from types import SimpleNamespace

"""
#################        Helper functions and labels          #################
"""

COCO_INSTANCE_CATEGORY_NAMES = [
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
    "backpack",
    "umbrella",
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
    "dining table",
    "toilet",
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
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def extract_predictions(predictions_, conf_thresh):
    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[int(i)] for i in list(predictions_["labels"])]
    #  print("\npredicted classes:", predictions_class)
    if len(predictions_class) < 1:
        return [], [], []
    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    # print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = conf_thresh
    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold]
    if len(predictions_t) == 0:
        return [], [], []

    # predictions in score order
    predictions_boxes = [predictions_boxes[i] for i in predictions_t]
    predictions_class = [predictions_class[i] for i in predictions_t]
    predictions_scores = [predictions_score[i] for i in predictions_t]
    return predictions_class, predictions_boxes, predictions_scores


def plot_image_with_boxes(img, boxes, pred_cls, title):
    plt.style.use("ggplot")
    text_size = 1
    text_th = 3
    rect_th = 1

    for i in range(len(boxes)):
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

    plt.figure()
    plt.axis("off")
    plt.title(title)
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    plt.show()


"""
#################        Evasion settings        #################
"""
eps = 8
eps_step = 2
max_iter = 5
batch_size = 1


"""
#################        Model definition        #################
"""
MODEL = "yolov3"  # OR yolov5 OR yolov8


if MODEL == "yolov3":
    print("yolov3 testing: ...")

    from mmdet.apis import init_detector, inference_detector
    from mmdet.registry import VISUALIZERS
    import mmcv
    from mmdet.utils.misc import get_test_pipeline_cfg
    from mmcv.transforms import Compose
    
    class MyYoloV3(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def compute_loss(self, imgs, targets, test_pipeline=None):
            model = self.model
            model.train()
            targets.gt_instances = targets.pred_instances
            
            if isinstance(imgs, (list, tuple)):
                is_batch = True
            else:
                imgs = [imgs]
                is_batch = False
            if isinstance(targets, (list, tuple)):
                targets = targets
            else:
                targets = [targets]
            
            cfg = model.cfg
            
            if test_pipeline is None:
                cfg = cfg.copy()
                test_pipeline = get_test_pipeline_cfg(cfg)
                if isinstance(imgs[0], np.ndarray):
                    # Calling this method across libraries will result
                    # in module unregistered error if not prefixed with mmdet.
                    test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

                test_pipeline = Compose(test_pipeline)        

            
            loss_list = []
            for i, img in enumerate(imgs):
                # prepare data
                if isinstance(img, np.ndarray) or isinstance(img, torch.Tensor):
                    # TODO: remove img_id.
                    data_ = dict(img=img, img_id=0)
                else:
                    # TODO: remove img_id.
                    data_ = dict(img_path=img, img_id=0)

                # build the data pipeline
                # data_ = test_pipeline(data_)

                # data_['inputs'] = [data_['inputs']]
                # data_['data_samples'] = [data_['data_samples']]

                # data = model.data_preprocessor(data_, training=False)
                # loss = model.loss(data['inputs'], targets)
                
                loss = model.loss(img, targets)
                parsed_losses, log_vars = self.model.parse_losses(loss) 
                loss_list.append(log_vars)

            if not is_batch:
                return loss_list[0]
            else:
                return loss_list
            
            
        def forward(self, batch_inputs, targets=None, y_mmdetection=None):
            if self.training:
                # 现在的问题就是怎么能拿到DetDataSample形式的targets
                # 最暴力的方法就是在所有需要传递普通targets的地方 
                # 都传递一个新参数targets_for_DetDataSample
                
                # targets应该来自 SingleStageDetector: predict()
                # targets list[:obj:`DetDataSample`]
                # batch_inputs: Input images of shape (N, C, H, W).
                # loss = self.model.loss(batch_inputs, batch_data_samples) #loss dict: A dictionary of loss components.
                loss_list = self.compute_loss(batch_inputs, targets=y_mmdetection)
                def extract_loss(losses):
                    if isinstance(losses, list):
                        loss = torch.stack([d.get('loss') for d in losses]) 
                    elif isinstance(losses, dict):
                        loss = losses.get('loss')
                        loss = loss.unsqueeze(0)  # 增加一个维度
                    else:
                        raise ValueError("Invalid input, must be a list of dictionaries or a dictionary")
                    loss.requires_grad_(True)
                    return loss
                loss = extract_loss(loss_list)
                # loss_components_dict = {"loss_total": loss_list['loss']}
                loss_components_dict = {"loss_total": loss}
                return loss_components_dict
            else:
                batch_inputs = batch_inputs.permute(0, 2, 3, 1).cpu().numpy()
                batch_inputs = (batch_inputs * 255).astype(np.uint8)  
 
                if batch_inputs.shape[0] == 1:
                    batch_inputs = batch_inputs.squeeze(0)
                else:
                    batch_inputs = [temp for temp in batch_inputs]
                mm_results = inference_detector(self.model, batch_inputs)
                if isinstance(mm_results, list):
                    pred_results = [result.pred_instances for result in mm_results]
                    results = []
                    for pred_result in pred_results:
                        result = {
                            "boxes": pred_result.bboxes,
                            "labels": pred_result.labels,
                            "scores": pred_result.scores,
                        }
                        results.append(result)
                    return results
                else:
                    pred_result = mm_results.pred_instances
                    result = {
                        "boxes": pred_result.bboxes,
                        "labels": pred_result.labels,
                        "scores": pred_result.scores,
                    }
                    return [result]
                
                
    config_file = '/home/jiawei/data/zjw/mmdetection/my_configs/yolov3_d53_8xb8-ms-608-273e_coco.py'
    checkpoint_file = '/home/jiawei/data/zjw/mmdetection/checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
    mmdet_model = init_detector(config_file, checkpoint_file, device='cuda:0')

    model = MyYoloV3(mmdet_model)

 
    detector = PyTorchYolo(
        model=model, model_type="yolov3", device_type="gpu", input_shape=(3, 608, 608), clip_values=(0, 255), attack_losses=("loss_total",)
    )

elif MODEL == "yolov5":
    print("testing yolov5 ...")
    import yolov5
    from yolov5.utils.loss import ComputeLoss

    matplotlib.use("TkAgg")

    class Yolo(torch.nn.Module):
        def __init__(self, model):
            print("\nYolov5 __init__\n")
            super().__init__()
            self.model = model
            self.model.hyp = {
                "box": 0.05,
                "obj": 1.0,
                "cls": 0.5,
                "anchor_t": 4.0,
                "cls_pw": 1.0,
                "obj_pw": 1.0,
                "fl_gamma": 0.0,
            }
            self.compute_loss = ComputeLoss(self.model.model.model)

        def forward(self, x, targets=None):
            print("\nYolov5 forward()\n")
            if self.training:
                outputs = self.model.model.model(x)
                loss, loss_items = self.compute_loss(outputs, targets)
                loss_components_dict = {"loss_total": loss}
                print("\nreturn loss_components_dict\n")
                return loss_components_dict
            else:
                print("\nreturn self.model(x)\n")
                return self.model(x)
    print("\nload yolov5ws.pt\n")
    model = yolov5.load("yolov5s.pt")
    print("\n model = Yolov5(model)\n")
    model = Yolo(model)
    print("\ndetector = PyTorchYolo(...)\n")
    detector = PyTorchYolo(
        model=model, model_type="yolov5", device_type="gpu", input_shape=(3, 640, 640), clip_values=(0, 255), attack_losses=("loss_total",)
    )
    
    
elif MODEL == "yolov8":
    print("testing for yolov8:")

    from ultralytics import YOLO  
    from ultralytics.utils import loss as yolov8loss
    from ultralytics.models.yolo.detect import DetectionTrainer
    
    class Yolov8(torch.nn.Module):
        def __init__(self, model):
            print("\nYolov8 __init__\n")
            super().__init__()
            self.model = model
            self.training = False
            args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
            self.model.trainer = DetectionTrainer(overrides=args)
            # args references https://docs.ultralytics.com/usage/cfg/#train
            self.model.model.args = SimpleNamespace(
                box=7.5,
                cls=0.5,
                dfl=1.5,
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.model.to(self.device)
            self.compute_loss = yolov8loss.v8DetectionLoss(self.model.model)
            
        def forward(self, x,  targets=None, y_mmdetection=None):
            print("\nYolov8 forward()\n")
            if self.model.model.training:
                outputs = self.model.model(x)
                loss, loss_items = self.compute_loss(outputs, targets)
                # loss, loss_items = yolov3_loss.compute_loss(outputs, targets, self.model.model)
                
                loss_box, loss_cls, loss_dfl = loss_items
                print(f"\n loss_box: {loss_box:.4f}")
                print(f"loss_cls: {loss_cls:.4f}")
                print(f"loss_dfl: {loss_dfl:.4f}\n")
                
                loss_items_dict = {"loss_total":loss}
                print("\nreturn loss_items_dict\n")
                return loss_items_dict
            else:
                print("\nreturn self.model(x)\n")
                return self.model(x)
        def train(self, mode=True):
          if mode:
              print("\n mode train \n")
              # If mode is True, call the original train method with the trainer
              self.model.trainer.train()
          else:
              print("\n mode eval \n")
              # If mode is False, set the model to evaluation mode
              self.model.eval()

            
    def move_model_to_device(model, device):
            for child in model.children():
                move_model_to_device(child, device)
            model.to(device)       


    
    
    print("loading yolov8n.pt...\n")
    model_yolov8 = YOLO('yolov8n.pt') # ultralytics

    
    print("model = Yolov8(model)\n")
    model = Yolov8(model_yolov8) # for art wrapper
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    move_model_to_device(model, device)
    
    
    print("art wrapper: detector = PyTorchYolo(...) \n")
    detector = PyTorchYolo(
        model=model, model_type="ultralytics_yolov8", device_type="gpu", input_shape=(3, 640, 640), clip_values=(0, 255), attack_losses=("loss_total",)
    )
    print("PyTorchYolo-detector.device()", detector._device)
    print("\n art wrapper done!!! \n")
"""
#################        Example image        #################
"""
# response = requests.get("https://ultralytics.com/images/zidane.jpg")
# img = np.asarray(Image.open(BytesIO(response.content)).resize((640, 640)))

# print("cv2.imwrite('ori_image.png'")
# print("Image shape:", img.shape)
# print("Image data type:", img.dtype)
# print("Image min:", img.min())
# print("Image max:", img.max())
# img_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imwrite('ori_image.png', img_save)



img_path = '/home/jiawei/data/zjw/images/10best-cars-group-cropped-1542126037.jpg'
# img_path = 'banner-diverse-group-of-people-2.jpg'
img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imwrite('ori_image1.png', img)
ori_shape = img.shape
print(ori_shape)

img = cv2.resize(img, dsize=(608, 608), interpolation=cv2.INTER_CUBIC)
img = np.asarray(img)
img_reshape = img.transpose((2, 0, 1))
image = np.stack([img_reshape], axis=0).astype(np.float32) 

x = image.copy()

# print("\n predict test yolov8\n ")
# result = model_yolov8.predict(img, show_labels=True, show=True)
# print("\n predict test yolov8 done...\n ")


print("art wrapper predict testing... \n")
threshold = 0.5  # 0.5(v5) or 0.85(v3)
dets = detector.predict(x)

preds = extract_predictions(dets[0], threshold)
plot_image_with_boxes(img=img, boxes=preds[1], pred_cls=preds[0], title="Predictions on original image")
print("\n art wrapper predict testing done!!! \n")
"""
#################        Evasion attack        #################
"""

print("\n Release cache...\n")
torch.cuda.empty_cache()
print("\n attack method is PGD...\n")
attack = ProjectedGradientDescent(estimator=detector, eps=eps, eps_step=eps_step, max_iter=max_iter, batch_size=batch_size)
print("Trying to generate adversarial image...\n")

x_mmdetction = x.transpose(0, 2, 3, 1)
x_mmdetction = (x_mmdetction * 255).astype(np.uint8)  

if x_mmdetction.shape[0] == 1:
    x_mmdetction = x_mmdetction.squeeze(0)
else:
    x_mmdetction = [temp for temp in x_mmdetction]
y_mmdetection = inference_detector(mmdet_model, x_mmdetction)
# y_mmdetection = None
image_adv = attack.generate(x=x, y=None,y_mmdetection=y_mmdetection)

###################################################################
# save adversarial_image 
image_to_save = image_adv[0].transpose(1,2,0).astype(np.uint8)
# image_to_save = cv2.resize(image_to_save, (ori_shape[1],ori_shape[0]))
# image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_BGR2RGB)
print("cv2.imwrite('adversarial_image.png'")
print("Image shape:", image_to_save.shape)
print("Image data type:", image_to_save.dtype)
print("Image min:", image_to_save.min())
print("Image max:", image_to_save.max())
# plt.imsave(
#     'adversarial_image2.png', image_to_save
# )
cv2.imwrite('adversarial_image1.png',image_to_save)
###################################################################
print("\nThe attack budget eps is {}".format(eps))
print("The resulting maximal difference in pixel values is {}.".format(np.amax(np.abs(x - image_adv))))

plt.axis("off")
plt.title("adversarial image")
plt.imshow(image_adv[0].transpose(1, 2, 0).astype(np.uint8), interpolation="nearest")
plt.show()

print("Original image shape:", x.shape)
print("Original image min:", x.min(), "max:", x.max())

# ... Attack generation ...

print("Adversarial image shape ", image_adv.shape)
print("Adversarial image min:", image_adv.min(), "max:", image_adv.max())





image_test = img.copy()
image_test += 1

# print("\n predict test yolov8 image_adv\n ")
# test = image_adv[0].transpose(1,2,0).astype(np.uint8)
# print(test.shape)
# plt.axis("off")
# plt.title("test image")
# plt.imshow(test, interpolation="nearest")
# plt.show()

# result = model_yolov8.predict(img, show_labels=True)
# print("\n predict test yolov8 image_adv done\n ")



# dets = detector.predict(image_adv)
# preds = extract_predictions(dets[0], threshold)
# plot_image_with_boxes(
#     img=image_adv[0].transpose(1, 2, 0).copy(),
#     boxes=preds[1],
#     pred_cls=preds[0],
#     title="Predictions on adversarial image",
# )
