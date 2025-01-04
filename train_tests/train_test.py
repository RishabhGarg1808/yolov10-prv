import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings("ignore")


from ultralytics.models import YOLOv10

model = YOLOv10("yolov10-custom.yaml")
model.train(data="sample/data.yaml",
            model = "yolov10-custom.yaml",
            epochs=50,
            pretrained=False,
            imgsz=32,
            name="yolov10-custom",
            simplify=False,
            )
