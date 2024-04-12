import cv2
import numpy as np
import matplotlib.pyplot as plt


from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import CarliniLInfMethod, CarliniL2Method # CW, Carlini & Wagner (C&W)
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.square_attack import SquareAttackDetection
from art.attacks.evasion import AutoAttack 
from art.attacks.evasion import SimBA #Simple Black-box Adversarial
import torchvision
import torch
from torchvision.models.detection.image_list import ImageList # to fix RegionProposalNetwork self.model.rpn error

from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
from typing import Optional, Tuple, Union, List, Dict

from torchvision.ops import boxes as box_ops, roi_align
import torch.nn.functional as F



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


class MyFasterRCNN(torch.nn.Module):
    def __init__(self,pretrained_model):
        super().__init__()
        self.model = pretrained_model
        
    def get_ori_images_size(self,images):
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        return original_image_sizes
    
    
    def postprocess_detections(
            self,
            class_logits,  
            box_regression, 
            proposals,  
            image_shapes, 
        ):

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.model.roi_heads.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        ##########################################################
        boxes_logits  = class_logits.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_logtis =[]
        for boxes, scores, image_shape, box_logits  in zip(pred_boxes_list, pred_scores_list, image_shapes, boxes_logits):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)


            # index for box_logits
            inds_box_logits = torch.where(scores > self.model.roi_heads.score_thresh)[0]
            
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            # box_logits = box_logits[:, 1:] #不需要移除背景label 

            
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            # box_logits = box_logits.reshape(-1)  #保留原来的维度
            
            # remove low scoring boxes
            inds = torch.where(scores > self.model.roi_heads.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            box_logits = box_logits[inds_box_logits] #(246,91)

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            box_logits = box_logits[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.model.roi_heads.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.model.roi_heads.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            box_logits = box_logits[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_logtis.append(box_logits)

        return all_boxes, all_scores, all_labels, all_logtis
    
    def forward(self, images, targets=None):
        original_image_sizes= self.get_ori_images_size(images)
        images, targets = self.model.transform(images, targets)
        features = self.model.backbone(images.tensors)
        # image_list = ImageList(images, [img.shape[-2:] for img in images])
        proposals, proposal_losses = self.model.rpn(images, features, targets)
        
        if self.training:
            self.model.train()
            detections, detector_losses = self.model.roi_heads(features, proposals, images.image_sizes, targets)
            detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            box_features = self.model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
            box_features = self.model.roi_heads.box_head(box_features)
            class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)
            
            _, _, _, target_class_logtis = self.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
            
            
            loss_objectness, loss_rpn_box_reg = proposal_losses["loss_objectness"], proposal_losses["loss_rpn_box_reg"] 
            loss_classifier, loss_box_reg = detector_losses["loss_classifier"],  detector_losses["loss_box_reg"]
            loss = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "none_detector_loss":None,
            }
            return loss
        else:
            detections, _ = self.model.roi_heads(features, proposals, images.image_sizes)
            detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            box_features = self.model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
            box_features = self.model.roi_heads.box_head(box_features)
            class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)
            
            _, _, _, target_class_logtis = self.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
            
            
            results = []
            for detection, box_logits in zip(detections, target_class_logtis):
                result = {
                    "boxes": detection["boxes"],
                    "labels": detection["labels"],
                    "scores": detection["scores"],
                    "class_logits": box_logits,
                }
                results.append(result)
            return results
    
    
def main():
    # Load the pre-trained Faster RCNN model.
    pretrained_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
    )
    # config_file = '/home/zjw/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = '/home/zjw/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # mmdet_model = init_detector(config_file, checkpoint_file, device='cuda:0')
  
    # mmdetection v3.x 测试单张图片并展示结果
    # img = '/home/zjw/data/images/demo.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
    # result = inference_detector(mmdet_model, img)

    # # init the visualizer(execute this block only once)
    # visualizer = VISUALIZERS.build(mmdet_model.cfg.visualizer)
    # # the dataset_meta is loaded from the checkpoint and
    # # then pass to the model in init_detector
    # visualizer.dataset_meta = mmdet_model.dataset_meta
    # img = mmcv.imread(img)
    # # show the results
    # visualizer.add_datasample(
    #     'result',
    #     img,
    #     data_sample=result,
    #     draw_gt=False,
    #     wait_time=0,
    #     out_file='outputs/result.png' # optionally, write to output file
    # )
    # visualizer.show()
    # Create an instance of MyFasterRCNN and pass the pre-trained model as an argument.
    model = MyFasterRCNN(pretrained_model)
    
    
    # Create ART object detector
    frcnn = PyTorchFasterRCNN(
        model=model, clip_values=(0, 255), 
        # attack_losses=[ "none_detector_loss"],
        attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
        attack_method='CW',
        input_shape = (3, 2139, 3500),
    )
    # frcnn = PyTorchFasterRCNN(
    #     clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    # )

    # Load image 1
    image_0 = cv2.imread("/home/jiawei/data/zjw/images/10best-cars-group-cropped-1542126037.jpg")
    image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)  # Convert to RGB
    print("image_0.shape:", image_0.shape)

    # Load image 2
    image_1 = cv2.imread("/home/jiawei/data/zjw/images/banner-diverse-group-of-people-2.jpg")
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_1 = cv2.resize(image_1, dsize=(image_0.shape[1], image_0.shape[0]), interpolation=cv2.INTER_CUBIC)
    print("image_1.shape:", image_1.shape)

    # Stack images
    image = np.stack([image_0, image_1], axis=0).astype(np.float32)
    print("image.shape:", image.shape)
    # image.shape: (2, 2139, 3500, 3) N H W C
    
    image = image.transpose(0,3,1,2)
    print("image.shape after transpose:", image.shape)
    
    for i in range(image.shape[0]):
        plt.axis("off")
        plt.title("image {}".format(i))
        plt.imshow(image[i].transpose(1,2,0).astype(np.uint8), interpolation="nearest")
        plt.show()

    # Make prediction on benign samples
    predictions = frcnn.predict(x=image)

    for i in range(image.shape[0]):
        print("\nPredictions image {}:".format(i))

        # Process predictions
        predictions_class, predictions_boxes, predictions_class = extract_predictions(predictions[i])

        # Plot predictions
        plot_image_with_boxes(img=image[i].transpose(1,2,0).astype(np.uint8).copy(), boxes=predictions_boxes, pred_cls=predictions_class)

    # Create and run attack
    print("\n Create and run attack ...")
    eps = 8
    attack = ProjectedGradientDescent(estimator=frcnn, eps=eps, eps_step=2, max_iter=10)
    attack2 = CarliniLInfMethod(estimator=frcnn)
    # attack3 = CarliniL2Method(estimator=frcnn, max_iter=5, max_halving=2, max_doubling=2)
    attack3 = CarliniL2Method(estimator=frcnn)
    attack4 = SquareAttackDetection(estimator=frcnn, norm=np.inf, max_iter=5000, eps=eps, p_init=0.8, nb_restarts=1)
    image_adv = attack4.generate(x=image, y=None)
    # image_adv = attack.generate(x=image, y=None)

    print("\nThe attack budget eps is {}".format(eps))
    print("The resulting maximal difference in pixel values is {}.".format(np.amax(np.abs(image - image_adv))))

    for i in range(image_adv.shape[0]):
        plt.axis("off")
        plt.title("image_adv {}".format(i))
        plt.imshow(image_adv[i].transpose(1,2,0).astype(np.uint8), interpolation="nearest")
        plt.show()

    predictions_adv = frcnn.predict(x=image_adv)

    for i in range(image.shape[0]):
        print("\nPredictions adversarial image {}:".format(i))

        # Process predictions
        predictions_adv_class, predictions_adv_boxes, predictions_adv_class = extract_predictions(predictions_adv[i])

        # Plot predictions
        plot_image_with_boxes(img=image_adv[i].transpose(1,2,0).copy(), boxes=predictions_adv_boxes, pred_cls=predictions_adv_class)


if __name__ == "__main__":
    main()



