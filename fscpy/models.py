from torchvision import models


class MyResNet(models.ResNet):
    """
	ResNet-18 model for 6 classes.
	"""

    def __init__(self):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=6)
