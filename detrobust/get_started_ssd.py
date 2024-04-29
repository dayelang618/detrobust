
import numpy as np
import torch


from art.attacks.evasion import ProjectedGradientDescent

from art.estimators.object_detection.pytorch_ssd import PytorchSSD
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent

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
eps_step = 1
max_iter = 10
batch_size = 2


"""
#################        Model definition        #################
"""


from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
from mmdet.utils.misc import get_test_pipeline_cfg
from mmcv.transforms import Compose

class MySSD(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def compute_loss(self, imgs, targets, test_pipeline=None):
            model = self.model
            model.train()

            
            if imgs.shape[0] > 1:
                is_batch = True
            else:
                imgs = [imgs]
                is_batch = False
            if isinstance(targets, (list, tuple)):
                for target in targets:
                    target.gt_instances = target.pred_instances
            else:
                targets = [targets]
                targets.gt_instances = targets.pred_instances     

            
            loss_list = []


            loss = model.loss(imgs, targets)
            parsed_losses, log_vars = self.model.parse_losses(loss) 
            loss_list.append(log_vars)

            if not is_batch:
                return loss_list[0]
            else:
                return loss_list
        
        
    def forward(self, batch_inputs, targets=None, y_mmdetection=None):
        if self.training:

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
                
                
config_file = '/home/jiawei/data/zjw/mmdetection/my_configs/ssd512_coco.py'
checkpoint_file = '/home/jiawei/data/zjw/mmdetection/checkpoints/ssd512_coco_20210803_022849-0a47a1ca.pth'
mmdet_model = init_detector(config_file, checkpoint_file, device='cuda:0')

model = MySSD(mmdet_model)


detector = PytorchSSD(
    model=model, device_type="gpu", input_shape=(3, 512, 512), clip_values=(0, 255), attack_losses=("loss_total",)
)


"""
#################        Example image        #################
"""



img_path = '/home/jiawei/data/zjw/images/10best-cars-group-cropped-1542126037.jpg'
img_path_1 = '/home/jiawei/data/zjw/images/banner-diverse-group-of-people-2.jpg'
img = cv2.imread(img_path)
img1 = cv2.imread(img_path_1)
ori_shape = img.shape
print(ori_shape)

img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
img = np.asarray(img)
img_reshape = img.transpose((2, 0, 1))

img1 = cv2.resize(img1, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
img1 = np.asarray(img1)
img1_reshape = img1.transpose((2, 0, 1))

image = np.stack([img_reshape,img1_reshape] , axis=0).astype(np.float32) 

x = image.copy()


print("art wrapper predict testing... \n")
threshold = 0.5  # 0.5(v5) or 0.85(v3)
dets = detector.predict(x=image)

# preds = extract_predictions(dets[0], threshold)
# plot_image_with_boxes(img=img, boxes=preds[1], pred_cls=preds[0], title="Predictions on original image")

print("\n art wrapper predict testing done!!! \n")
"""
#################        Evasion attack        #################
"""

print("\n Release cache...\n")
torch.cuda.empty_cache()
print("\n attack method is PGD...\n")
attack = ProjectedGradientDescent(estimator=detector, eps=eps, eps_step=eps_step, max_iter=max_iter, batch_size=batch_size)

# attack = AutoProjectedGradientDescent(estimator=detector, eps=eps, eps_step=eps_step, 
#                                         max_iter=100, targeted=False, nb_random_init=1,
#                                         batch_size=batch_size, loss_type=None, )
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
cv2.imwrite('adversarial_image_ssd_cw.png',image_to_save)
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

