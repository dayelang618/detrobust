import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from art.estimators.object_detection.pytorch_ssd import PytorchSSD
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms


from mmdet.apis import init_detector, inference_detector


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
            scale_w = model_input_size / ori_image_size[1]
            scale_h = model_input_size / ori_image_size[0]
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


def generate_adversarial_image(image, attack):
    x_mmdetction = image.transpose(0, 2, 3, 1)
    x_mmdetction = (x_mmdetction * 255).astype(np.uint8)  

    if x_mmdetction.shape[0] == 1:
        x_mmdetction = x_mmdetction.squeeze(0)
    else:
        x_mmdetction = [temp for temp in x_mmdetction]
    y_mmdetection = inference_detector(mmdet_model, x_mmdetction)
    return attack.generate(x=image, y=None,y_mmdetection=y_mmdetection)
    # try:
    #     return attack.generate(x=image, y=None,y_mmdetection=y_mmdetection)
    # except IndexError:
    #     print("\n ########### skip this iteration ########### \n")
    #     return None  
#################        Evasion settings        #################
adversarial_save = True


eps = 8 # 8
eps_step = 1
max_iter = 10
batch_size = 4
#################        Model Wrapper       #################
model_input_size = 512
config_file = '/home/jiawei/data/zjw/mmdetection/my_configs/ssd512_coco.py'
checkpoint_file = '/home/jiawei/data/zjw/mmdetection/checkpoints/ssd512_coco_20210803_022849-0a47a1ca.pth'


attack_method = 'PGD'
mmdet_model = init_detector(config_file, checkpoint_file, device='cuda:0')
model = MySSD(mmdet_model)

detector = PytorchSSD(
    model=model, device_type="gpu", input_shape=(3, model_input_size, model_input_size), clip_values=(0, 255), attack_losses=("loss_total",)
)
if attack_method == 'PGD':
    print("\n attack method is PGD...\n")
    attack = ProjectedGradientDescent(estimator=detector, eps=eps, eps_step=eps_step, max_iter=max_iter, batch_size=batch_size)
elif attack_method == 'APGD':
     print("\n attack method is APGD...\n")
     attack = AutoProjectedGradientDescent(estimator=detector, eps=eps, eps_step=eps_step, 
                                           max_iter=max_iter, targeted=False, nb_random_init=1,
                                           batch_size=batch_size, loss_type=None, )
#################        Load COCO dataset      #################
image_transform = transforms.Compose([
    transforms.Resize((model_input_size, model_input_size)),
])
dataDir = '/home/jiawei/data/zjw/datasets/coco/images/val2017'
annFile='/home/jiawei/data/zjw/datasets/coco/annotations/instances_val2017.json'
# dataDir = '/home/jiawei/data/zjw/datasets/coco_small/train_2017_small'
# annFile='/home/jiawei/data/zjw/datasets/coco_small/instances_train2017_small.json'
dataset = CocoDetection(root=dataDir,annFile=annFile, transform=image_transform)


dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=dataset.collate_fn)

output_directory = "output_adv_images_ssd_cw"
os.makedirs(output_directory, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.4
#################        Attack Loop       #################
for iter_num, (images, targets, image_filenames) in enumerate(dataloader):
    batch_num = iter_num

    print("\n ################################ iter_num = ", iter_num, " #########################\n")
    mmdet_model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model = MySSD(mmdet_model) # for art wrapper

    detector = PytorchSSD(
        model=model, device_type="gpu", input_shape=(3, model_input_size, model_input_size), clip_values=(0, 255), attack_losses=("loss_total",)
    )
    
    if attack_method == 'PGD':
        print("\n attack method is PGD...\n")
        attack = ProjectedGradientDescent(estimator=detector, eps=eps, eps_step=eps_step, max_iter=max_iter, batch_size=batch_size)
    elif attack_method == 'APGD':
        print("\n attack method is APGD...\n")
        attack = AutoProjectedGradientDescent(estimator=detector, eps=eps, eps_step=eps_step, 
                                            max_iter=max_iter, targeted=False, nb_random_init=1,
                                            batch_size=batch_size, loss_type=None, )
    
    images = images.mul(255).byte().numpy()
    
    
    if not adversarial_save:
        # yolov3 inference
        # result = inference_detector(mmdet_model, images)
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
            # image_name =  f"adv_image_{i+1+batch_num*16}.png"
            original_filename = image_filenames[i]
            base_name, ext = os.path.splitext(original_filename)
            output_path = os.path.join(output_directory, base_name + '.jpg')
            cv2.imwrite(output_path, image_save)
        


        
        