import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from types import SimpleNamespace

from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import ProjectedGradientDescent
import sys
sys.path.append("/home/jiawei/data/zjw/ultralytics")
from ultralytics import YOLO  
from ultralytics.utils import loss as yolov8loss
from ultralytics.models.yolo.detect import DetectionTrainer

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F 
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torchvision import transforms


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
    rect_th = 3

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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.axis("off")
    plt.title(title)
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    plt.show()
    
    
#################        COCO Dataset            #################
class CocoDetection(Dataset):
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
    



#################        Model definition        #################

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
    def forward(self, x,  targets=None):
        print("\nYolov8 forward()\n")
        if self.model.model.training:
            outputs = self.model.model(x)
            loss, loss_items = self.compute_loss(outputs, targets)
            loss_items_dict = {"loss_total":loss}
            print("\nYolov8 return loss_items_dict\n")
            return loss_items_dict
        else:
            print("\nYolov8 return self.model(x)\n")
            return self.model(x)
    def train(self, mode=True):
        if mode:
            # If mode is True, call the original train method with the trainer
            self.model.trainer.train()
        else:
            # If mode is False, set the model to evaluation mode
            self.model.eval()


def generate_adversarial_image(image, attack):
    try:
        return attack.generate(x=image, y=None)
    except IndexError:
        print("\n ########### skip this iteration ########### \n")
        return None  
#################        Evasion settings        #################

eps = 8 # 8
eps_step = 2
max_iter = 5
batch_size = 16
#################        Model Wrapper       #################

model_yolov3 = YOLO('yolov3u.pt') # ultralytics
model = Yolov8(model_yolov3) # for art wrapper

detector = PyTorchYolo(
    model=model, model_type="ultralytics_yolov8", device_type="gpu", input_shape=(3, 640, 640), clip_values=(0, 255), attack_losses=("loss_total",)
)
attack = ProjectedGradientDescent(
    estimator=detector,
    eps=eps, eps_step=eps_step, max_iter=max_iter, 
    batch_size=batch_size)

#################        Load COCO dataset      #################
transform = transforms.Compose([
    transforms.Resize((640, 640)),
])
dataDir = '/home/jiawei/data/zjw/datasets/coco/images/val2017'
annFile='/home/jiawei/data/zjw/datasets/coco/annotations/instances_val2017.json'
# dataDir = '/home/jiawei/data/zjw/datasets/coco_small/train_2017_small'
# annFile='/home/jiawei/data/zjw/datasets/coco_small/instances_train2017_small.json'
dataset = CocoDetection(root=dataDir,annFile=annFile, transform=transform)


train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=dataset.collate_fn)

output_directory = "output_adv_images/test"
os.makedirs(output_directory, exist_ok=True)
#################        Attack Loop       #################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adversarial_save = True
adversarial_inference = False

threshold = 0.4
for iter_num, (images, targets, image_filenames) in enumerate(train_dataloader):
    batch_num = iter_num
    # image_filenames = sorted(image_filenames)

    print("\n ################################ iter_num = ", iter_num, " #########################\n")
    model_yolov3 = YOLO('yolov3u.pt') # ultralytics
    model = Yolov8(model_yolov3) # for art wrapper

    detector = PyTorchYolo(
        model=model, model_type="ultralytics_yolov8", device_type="gpu", input_shape=(3, 640, 640), clip_values=(0, 255), attack_losses=("loss_total",)
    )
    attack = ProjectedGradientDescent(
        estimator=detector,
        eps=eps, eps_step=eps_step, max_iter=max_iter, 
        batch_size=batch_size)
    
    images = images.mul(255).byte().numpy()
    # images = (images*255).numpy().astype(np.uint8)
    
    
    if adversarial_save != True:
        # yolov8 inference
        result = model_yolov3.predict(cv2.cvtColor(images, cv2.COLOR_RGB2BGR), show_labels=True, show=True)
        # ART wrapper inference
        dets = detector.predict(images)
        preds = extract_predictions(dets[0], threshold)
        img = images[0].transpose(1,2,0).copy()
        plot_image_with_boxes(img=img, boxes=preds[1], 
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
    
    if adversarial_save != True:
    # show adversarial images (batch size > 1)
        for i in range(image_adv.shape[0]):
            plt.figure()
            plt.axis("off")
            plt.title("adversarial image")
            plt.imshow(image_adv[i].transpose(1, 2, 0).astype(np.uint8), interpolation="nearest")
            plt.show()
        
    
    # ... Save images_adv ... (batch size > 1)
    if adversarial_save == True:
        for i in range(image_adv.shape[0]):
            image_save = image_adv[i].transpose(1,2,0).astype(np.uint8)
            image_save = cv2.cvtColor(image_save, cv2.COLOR_BGR2RGB)
            # image_name =  f"adv_image_{i+1+batch_num*16}.png"
            original_filename = image_filenames[i]
            base_name, ext = os.path.splitext(original_filename)
            output_path = os.path.join(output_directory, base_name + '.png')
            cv2.imwrite(output_path, image_save)
        

#################        Inference Loop       #################
    if adversarial_inference == True:
        model_yolov3 = YOLO('yolov3u.pt') # ultralytics
        for i in range(image_adv.shape[0]):
            image_adv_test = image_adv[i].transpose(1,2,0).astype(np.uint8)
            result = model_yolov3.predict(cv2.cvtColor(image_adv_test, cv2.COLOR_RGB2BGR), show_labels=True)

        
        
