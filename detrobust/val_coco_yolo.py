import torch
import sys
sys.path.append("/home/jiawei/data/zjw/ultralytics")
from ultralytics import YOLO
# Model loading
# model = YOLO('yolov8n.pt')  
model = YOLO('yolov3.pt')  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inference
# results = model.predict(source=images_path, conf=0.25, iou=0.5, device=device, half=True) 

# val small test
# images_path = '/home/jiawei/data/zjw/datasets/coco_samll_converted/images/train2017_small'  
# metrics = model.val(data='/home/jiawei/data/zjw/yaml/small_test.yaml', save_json=True)


metrics = model.val(data='/home/jiawei/data/zjw/yaml/coco-val-clean.yaml', save_json=True,
                            imgsz=640,
                            device='0')

print("map50-95: ", metrics.box.map)    # map50-95
print("map50: ", metrics.box.map50)  # map50
print("map75: ", metrics.box.map75)  # map75

print("\n metrics: ", metrics.box.maps)