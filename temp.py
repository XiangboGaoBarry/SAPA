from utils.utils import PCA_fit_imagenet, to_device
from torchvision.models import resnet50, resnet18, resnet101
classifier = to_device(resnet50(pretrained=True))
classifier.eval()
PCA_fit_imagenet("/home/bar/xb/dataset/imagenet/val", classifier=None)