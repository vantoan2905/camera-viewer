from head import DetectionHead
from neck import Neck
from backbone import Backbone
from yolo_net import YOLONet



class TrainingYOLONet(YOLONet):
    def __init__(self, num_classes=20, num_anchors=3):
        super(TrainingYOLONet, self).__init__(num_classes, num_anchors)
        
        