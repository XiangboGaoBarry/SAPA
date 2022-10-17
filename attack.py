from models.PQGAN_attacker import PQGAN_attacker
import argparse
import os
from PIL import Image
import torch
from torchvision.models import resnet50
from torchvision.transforms import ToTensor, Normalize, ToPILImage
from utils.depth_prediction import DepthPrediction
from utils.rain_synthesis import RainSynthesisDepth
import numpy as np
import time
import pdb
import torch.nn.functional as F
from tqdm import tqdm
import glob


normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
inv_normalize = Normalize(mean=[1/0.485, 1/0.456, 1/0.406],
                                std=[1/0.229, 1/0.224, 1/0.225])

def save_adv_img(img, save_dir, name):
    img.shape
    to_save = torch.clamp(img * 255,0,255).to(torch.uint8).squeeze()
    to_save = ToPILImage()(to_save)
    # img_dir, img_name = img_path.split(os.sep)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_path = os.path.join(save_dir, name)
    to_save.save(save_path)
    
def center_crop(t):
    return t[:, :, 16:-16, 16:-16]

parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

parser.add_argument('--data_path', type=str, default="test_imgs")
parser.add_argument('--load_path', type=str, default="models/pretrained/nz128")
parser.add_argument('--save_path', type=str, default="attack_results")
parser.add_argument('--load_model', type=str, default="True")
parser.add_argument('--load_iter', type=int, default=19000)
parser.add_argument('--load_iter_1', type=int, default=0)
parser.add_argument('--load_iter_2', type=int, default=0)
parser.add_argument('--load_iter_34', type=int, default=0)
parser.add_argument('--image_size', type=tuple, default=(256, 256))

# architecture setup
parser.add_argument('--cond_channels', type=int, default=4)
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--nzp', type=int, default=128)
parser.add_argument('--nzp3', type=int, default=128)
parser.add_argument('--ndf', type=int, default=32)

# attack setup
parser.add_argument('--lr', type=float, default=0.1, help='learning rate used for adversarial attack')
parser.add_argument('--cuda', type=str, default='True', help='Availability of cuda')
parser.add_argument('--iters', type=int, default=10, help='number of attack iterations')

args = parser.parse_args()
args.cuda = True if args.cuda == 'True' else False
args.load_model = True if args.load_model == 'True' else False
if args.load_iter:
    args.load_iter_1 = args.load_iter_2 = args.load_iter_34 = args.load_iter

to_device = lambda x: x.cuda() if args.cuda else x.cpu()


with open(os.path.join(args.data_path, 'labels.txt'), 'r') as lf:
    labels = dict()
    line = lf.readline()
    while line:
        name, label = line.split(' ')
        labels[name] = int(label)
        line = lf.readline()



classifier = to_device(resnet50(pretrained=True))
classifier.eval()
attacker = PQGAN_attacker(batch_size=1, 
                         img_size=args.image_size, 
                         patch_size=64, 
                         nz=128, 
                         args=args, 
                         pattern_size=args.image_size)
depth_predicter = DepthPrediction()
rain_synthesizer = RainSynthesisDepth(alpha=0.002, beta=0.01, r_r=2, a=0.9)



for idx, filename in enumerate(os.listdir(args.data_path)):
    if not filename.endswith((".jpg", ".JPG", ".jpeg", ".JPEG", ".png")):
        continue
    gt = labels[filename]
    img = Image.open(os.path.join(args.data_path, filename))
    img = img.resize(args.image_size)
    img_np = np.array(img)
    img = to_device(ToTensor()(img).unsqueeze(0))
    with torch.no_grad():
        depth = depth_predicter(to_device(depth_predicter.midas_transforms(img_np)), args.image_size).detach()
            
    # attacker.init_para(cond_limit=[[1,1,1,1],[0,0,0,0]], cond_fix=[0.5,1,0.5,0.5])
    attacker.init_para(cond_limit=[[1,1,1,1],[0,0,0,0]], cond_fix=[0.5,1,0.5,0.5])
    attacker.set_atk_para(iterations=args.iters)
    # syn_img = img
    # outputs = classifier(center_crop(normalize(syn_img)))
    # pred_clean = torch.argmax(outputs)
    # pdb.set_trace()
    pbar = tqdm(range(args.iters), total=args.iters)
    for iter_i in pbar:
        rain_pattern = attacker.generate_pattern()
        syn_img = rain_synthesizer.synthesize(img, depth, rain_pattern)
        syn_img = syn_img.clamp(0,1)
        outputs = classifier(center_crop(normalize(syn_img)))
        prediction = torch.argmax(outputs)
        adv_loss = -F.cross_entropy(outputs, to_device(torch.Tensor([gt]).long()))
        attacker.step(adv_loss)
        # pbar.set_description("Image %02d | Iter %02d | Adv loss=%.3f | gt: %03d | pred clean: %03d | pred: %03d" % \
        #                     (idx, iter_i, adv_loss, gt, pred_clean, prediction))
        pbar.set_description("Image %02d | Iter %02d | Adv loss=%.3f | gt: %03d | pred: %03d | status: %s" % \
                            (idx, iter_i, adv_loss, gt, prediction, "SUCC" if prediction != gt else "FAILD"))
    save_adv_img(syn_img, args.save_path, filename)
        
    
    
