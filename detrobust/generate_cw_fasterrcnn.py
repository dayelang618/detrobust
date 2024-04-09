import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.object_detection import PyTorchFasterRCNN
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F 
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torchvision import transforms
import torchvision
from torchvision.ops import boxes as box_ops


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


def extract_predictions(predictions_, conf_thresh):
    # Get the predicted class
    try:
        predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[int(i)] for i in list(predictions_["labels"])]
        print("\npredicted classes:", predictions_class)
    except IndexError:
        pass
        
    if len(predictions_class) < 1:
        return [], [], []
    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    print("predicted score:", predictions_score)

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
    rect_th = 3
    
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
    plt.figure()
    plt.axis("off")
    plt.title(title)
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    plt.show()

    
    
#################        COCO Dataset            #################
class MyCocoDetection(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        
    """
    def __init__(self,root,annFile,transform=None, target_transform=None):
        super().__init__()
        self.root = root 
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = root # image folder

        COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
        self.label_map = COCO_LABEL_MAP
        self.batch_count = 0
        
    
    def __getitem__(self,index):
        coco = self.coco
        img_id = self.ids[index]
        #==============
        # image
        # =============
        path = coco.loadImgs(img_id)[0]['file_name']
        file_name = path
        img = Image.open(os.path.join(self.root,path)).convert('RGB')
        img = transforms.ToTensor()(img)
        # Pad to square resolution
        c, h, w = img.shape
    
    
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding（左，右，上，下）
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=0)
        _, padded_h, padded_w = img.shape

        #==============
        # labels
        # =============
        annids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annids)
        
        # Transform
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
            
        bboxes = []
        for i in range(len(anns)):
            bbox = [self.label_map[anns[i]['category_id']]-1]
            bbox.extend(anns[i]['bbox']) # (x,y,w,h) x和y表示bbox左上角的坐标，w和h表示bbox的宽度和高度
            bboxes.append(bbox)
            
        if bboxes: # Only if there are bounding boxes 
            bboxes = torch.from_numpy(np.array(bboxes))
            # Extract coordinates for unpadded + unscaled image（这好像计算出来的是bbox左上和右下两点的坐标）
            x1 = (bboxes[:, 1])
            y1 = (bboxes[:, 2])
            x2 = (bboxes[:, 1] + bboxes[:, 3])
            y2 = (bboxes[:, 2] + bboxes[:, 4])
            # Adjust for added padding（调整padding后两点的坐标）
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)（重新归一化，（x,y）表示中心点坐标，（w,h）表示bbox的宽和高）
            bboxes[:, 1] = ((x1 + x2) / 2) / padded_w
            bboxes[:, 2] = ((y1 + y2) / 2) / padded_h
            bboxes[:, 3] *= 1 / padded_w
            bboxes[:, 4] *= 1 / padded_h

            #bboxes的格式为(category,x,y,w,h)
            targets = torch.zeros((len(bboxes), 6))
            targets[:, 1:] = bboxes
        else:
            targets = None

        return img, targets, file_name
    
    
    def collate_fn(self, batch):
        """将数据和标签拼接成batch"""
        imgs, targets, filenames = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [bboxes for bboxes in targets if bboxes is not None]
        # Add sample index to targets
        for i, bboxes in enumerate(targets):
            bboxes[:, 0] = i # 使用索引表示哪些bboxes对应batch中的那张图片 此时bboxes的格式为(index,category,x,y,w,h)
        targets = torch.cat(targets, 0) #拼接
    
        imgs = torch.stack([img for img in imgs])
        self.batch_count += 1

        return imgs, targets, filenames

    def __len__(self):
        return len(self.ids)
    




#################        COCO Dataset            #################
class TestCocoDetection(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        
    """
    def __init__(self,root,annFile,transform=None, target_transform=None):
        super().__init__()
        self.root = root 
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = root # image folder

        COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
        self.label_map = COCO_LABEL_MAP
        self.batch_count = 0
        
    
    def __getitem__(self,index):
        coco = self.coco
        img_id = self.ids[index]
        #==============
        # image
        # =============
        path = coco.loadImgs(img_id)[0]['file_name']
        file_name = path
        img = Image.open(os.path.join(self.root,path)).convert('RGB')
        img = transforms.ToTensor()(img)
        # Pad to square resolution
        ori_image_size = img.shape
    

        #==============
        # labels
        # =============
        annids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annids)
        bboxes = []
        for i in range(len(anns)):
            bbox = [self.label_map[anns[i]['category_id']]-1]
            bbox.extend(anns[i]['bbox']) # (x,y,w,h) x和y表示bbox左上角的坐标，w和h表示bbox的宽度和高度
            bboxes.append(bbox)
        if bboxes: # Only if there are bounding boxes 
            bboxes = torch.from_numpy(np.array(bboxes))
            # Extract coordinates for unpadded + unscaled image
            x1 = (bboxes[:, 1])
            y1 = (bboxes[:, 2])
            x2 = (bboxes[:, 1] + bboxes[:, 3])
            y2 = (bboxes[:, 2] + bboxes[:, 4])
                # 计算缩放比例
            scale_w = 640 / ori_image_size[1]
            scale_h = 640 / ori_image_size[0]
            # 计算缩放后的宽和高
            new_w = bboxes[:,3] * scale_w
            new_h = bboxes[:,4] * scale_h

            # 计算缩放后的中心点
            x1 = bboxes[:,1] + 0.5 * bboxes[:,3]
            y1 = bboxes[:,2] + 0.5 * bboxes[:,4]
            new_x1 = x1 * scale_w
            new_y1 = y1 * scale_h

            # 计算缩放后的左上角坐标
            bboxes[:, 1] = new_x1
            bboxes[:, 2] = new_y1
            bboxes[:, 3] = new_w
            bboxes[:, 4] = new_h

            #bboxes的格式为(category,x,y,w,h)
            targets = torch.zeros((len(bboxes), 6))
            targets[:, 1:] = bboxes
        else:
            targets = None

                # Transform
        if self.transform:
            img = self.transform(img)

            
        return img, targets, file_name, ori_image_size
    
    
    def collate_fn(self, batch):
        """将数据和标签拼接成batch"""
        imgs, targets, filenames, ori_image_size = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [bboxes for bboxes in targets if bboxes is not None]
        # Add sample index to targets
        for i, bboxes in enumerate(targets):
            bboxes[:, 0] = i # 使用索引表示哪些bboxes对应batch中的那张图片 此时bboxes的格式为(index,category,x,y,w,h)
        targets = torch.cat(targets, 0) #拼接
    
        imgs = torch.stack([img for img in imgs])
        self.batch_count += 1

        return imgs, targets, filenames, ori_image_size

    def __len__(self):
        return len(self.ids)
#################        Model definition        #################
# Create ART object detector
# frcnn = PyTorchFasterRCNN(
#     clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
#     device_type='gpu'
# )
pretrained_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
)
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
    
    def cw_cls_loss(self, class_logits, num_classes=91, targeted=False, kappa=0):
        batch_losses = []
        for logits in class_logits:
            logits = torch.Tensor(logits)
            logits = 1 / 2 * (torch.tanh(logits) + 1)
            
            num_targets = logits.size(0)

            # 找到每个目标的最大logits及其索引
            max_logits, max_indices = torch.max(logits, dim=1)

            # 构造一个one-hot编码的目标标签张量
            one_hot_labels = F.one_hot(max_indices, num_classes=num_classes).float()

            # 计算除目标类别之外的其他类别的最大logit值
            other = torch.max((1 - one_hot_labels) * logits, dim=1)[0]

            # 计算目标类别的logit值
            real = torch.max(one_hot_labels * logits, dim=1)[0]

            # 计算f函数
            if targeted:
                f = torch.clamp(other - real, min=-kappa)
            else:
                f = torch.clamp(real - other, min=-kappa)

            cw_loss = f.sum()
            batch_losses.append(cw_loss / num_targets) #均值 求和太大了


        return sum(batch_losses) / len(class_logits)
    
    
    def forward(self, images, targets=None):
        original_image_sizes= self.get_ori_images_size(images)
        images, targets = self.model.transform(images, targets)
        features = self.model.backbone(images.tensors)
        # image_list = ImageList(images, [img.shape[-2:] for img in images])
        proposals, proposal_losses = self.model.rpn(images, features, targets)
        
        if self.training:
            self.model.train()
            
            # proposals, _, labels, regression_targets = self.model.roi_heads.select_training_samples(proposals, targets)
            box_features = self.model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
            box_features = self.model.roi_heads.box_head(box_features)
            class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)
            
            
            _, detector_losses = self.model.roi_heads(features, proposals, images.image_sizes, targets)

            
            target_boxes, target_scores, target_labels, target_class_logtis = self.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
            
            
            loss_objectness, loss_rpn_box_reg = proposal_losses["loss_objectness"], proposal_losses["loss_rpn_box_reg"] 
            #原来的cls loss不要了
            loss_classifier, loss_box_reg = detector_losses["loss_classifier"],  detector_losses["loss_box_reg"] 
        
            
            loss_classifier = self.cw_cls_loss(target_class_logtis)
            # print("\n cw_cls_loss: ",loss_classifier)
            loss = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "none_detector_loss":None,
            }
            # for loss_name, loss_value in loss.items():
            #     print(f"{loss_name}: {loss_value}")
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
model = MyFasterRCNN(pretrained_model)
frcnn = PyTorchFasterRCNN(
    model=model, clip_values=(0, 255), attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
    attack_method='CW'
)

def generate_adversarial_image(image, attack):
    try:
        return attack.generate(x=image, y=None)
    except IndexError:
        print("\n ########### skip this iteration ########### \n")
        return None  
#################        Evasion settings        #################
adversarial_save = True
adversarial_inference = False

eps = 8 # 8/255
eps_step = 2
max_iter = 5
batch_size = 8
#################        Model Wrapper       #################


attack = ProjectedGradientDescent(
    estimator=frcnn,
    eps=eps, eps_step=eps_step, max_iter=max_iter, 
    batch_size=batch_size)

#################        Load COCO dataset      #################
image_transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Resize((1333, 800)),#h,w 跟mmdetevtion cv2一致
])
dataDir = '/home/jiawei/data/zjw/datasets/coco/images/val2017'
annFile='/home/jiawei/data/zjw/datasets/coco/annotations/instances_val2017.json'
# dataDir = '/home/jiawei/data/zjw/datasets/coco_small/train_2017_small'
# annFile = '/home/jiawei/data/zjw/datasets/coco_small/instances_train2017_small.json'

# dataset = MyCocoDetection(root=dataDir,annFile=annFile, transform=image_transform)
dataset = TestCocoDetection(root=dataDir,annFile=annFile,transform=image_transform)


train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=dataset.collate_fn)


#################        Attack Loop       #################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


output_directory = "output_adv_images"
os.makedirs(output_directory, exist_ok=True)
threshold = 0.4
for iter_num, (images, targets, image_filenames, ori_image_size) in enumerate(train_dataloader):
    batch_num = iter_num
    ori_w_batch = [size[1] for size in ori_image_size]
    ori_h_batch = [size[2] for size in ori_image_size]
    print("\n ################################ iter_num = ", iter_num, " #########################\n")

    images = images.mul(255).byte().numpy()
    # images = (images*255).numpy().astype(np.uint8)
    
    if not adversarial_save:
        # ART wrapper inference
        dets = frcnn.predict(images)
        for i in range(images.shape[0]):
            print("\nPredictions image {}:".format(i))
            preds = extract_predictions(dets[i], threshold)
            # print("images[i].shape", images[i].shape)
            img = images[i].transpose(1,2,0).copy()
            plot_image_with_boxes(img=img,
                                    boxes=preds[1], 
                                    pred_cls=preds[0], 
                                    title="ART warrper detectors Predictions on original image")
        
    # generating adversarial image
    print("\n generating adversarial images... \n")
    
    image_adv = generate_adversarial_image(images, attack)
    if image_adv is None:
       print("iteration: ", batch_num)
       continue

    
    print("\nThe attack budget eps is {}".format(eps))
    print("The resulting maximal difference in pixel values is {}.".format(np.amax(np.abs(images- image_adv))))
    
    if not adversarial_save:
    # show adversarial images (batch size > 1)
        for i in range(image_adv.shape[0]):
            plt.figure()
            plt.axis("off")
            plt.title("adversarial image")
            plt.imshow(image_adv[i].transpose(1, 2, 0).astype(np.uint8), interpolation="nearest")
            plt.show()
        
    
    # ... Save images_adv ... (batch size > 1)
    if adversarial_save:

        for i in range(image_adv.shape[0]):
            image_save = image_adv[i].transpose(1,2,0).astype(np.uint8)
            image_save = cv2.cvtColor(image_save, cv2.COLOR_BGR2RGB)
            ori_w = ori_w_batch[i]
            ori_h = ori_h_batch[i]
            image_save_resize = cv2.resize(image_save, (ori_h, ori_w))
            
            original_filename = image_filenames[i]
            base_name, ext = os.path.splitext(original_filename)
            output_path = os.path.join(output_directory, base_name + '.jpg')
            cv2.imwrite(output_path, image_save)
        
    if adversarial_inference:
        dets = frcnn.predict(image_adv)
        # labels array([ 1,  2,  3,  4, 87, 85,  1, 16, 62]) 大于 80 报错
        for i in range(images.shape[0]):
            print("\nPredictions image {}:".format(i))

            preds = extract_predictions(dets[i], threshold)
            img = images[i].transpose(1,2,0).copy()
            plot_image_with_boxes(img=img, boxes=preds[1], 
                            pred_cls=preds[0], 
                            title=f"ART warrper detectors Predictions on adversarial image\n eps={eps}/255")

        
        
