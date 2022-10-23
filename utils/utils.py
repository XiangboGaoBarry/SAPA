from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def save_images_from_torch(images, dir):
    for i, img in enumerate(images):
        print(img.shape)
        np_img = img.cpu().detach().numpy().transpose((1,2,0))
        img = Image.fromarray(np.uint8(np_img)*255)
        img.save("%s/%d%s" % (dir, i, '.jpg'))

def show_images_from_torch(images):
    for i, img in enumerate(images):
        print(img.shape)
        np_img = img.cpu().detach().numpy().transpose((1,2,0))
        plt.imshow(np_img)
        plt.show()

def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


to_tensor = lambda shape: transforms.Compose([transforms.Resize(shape), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

def center_crop(t, batched=False):
    if batched:
        return t[:, :, 16:-16, 16:-16]
    else:
        return t[:, 16:-16, 16:-16]


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

# def get_adv_loss(classifier, x, y, targeted=False, kappa=-np.inf):
#     x = torch.cat([to_device(preprocess(center_crop(img)).unsqueeze(0)) for img in x])
#     outputs = classifier(x)
#     one_hot_labels = to_device(torch.eye(len(outputs[0]))[y])
#     i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
#     j = torch.masked_select(outputs, one_hot_labels.bool())
#     if targeted:
#         diff = i - j
#         stop_attack = diff > kappa
#         return (stop_attack * diff), diff
#     diff = j - i
#     stop_attack = diff > kappa
#     return (stop_attack * diff), diff


def normalize(z, to01=False):
    if to01:
        max = z.max()
        min = z.min()
        return (z - min) / (max - min)
    mean = z.mean()
    std = z.std()
    return (z - mean) / (std + 1e-7)



class ContentLoss:

    def __init__(self, model):
        self.model = model
        self.layer_index = [2, 4, 5, 6, 7]
        self.mse_loss = nn.MSELoss()

    def set_feature1(self, image):
        x = image.unsqueeze(0)
        self.feature1 = []
        seq_model = nn.Sequential(*list(self.model.children()))
        for i in range(len(seq_model)):
            if isinstance(seq_model[i],nn.Linear):
                break
            x = seq_model[i](x)
            if i in self.layer_index:
                feature = x.detach()
                feature.requires_grad = False
                self.feature1.append(feature)

    def get_loss(self, image):
        x = image
        feature2 = []
        seq_model = nn.Sequential(*list(self.model.children()))
        for i in range(len(seq_model)):
            if isinstance(seq_model[i],nn.Linear):
                break
            x = seq_model[i](x)
            if i in self.layer_index:
                feature2.append(x)

        loss_list = []
        for idx in range(len(self.feature1)):
            f1 = self.feature1[idx]
            f2 = feature2[idx]
            loss_list.append(self.mse_loss(f1, f2))
        return loss_list


def get_adv_loss(classifier, x, y, targeted=False, kappa=-np.inf):
    x = torch.cat([to_device(preprocess(center_crop(img)).unsqueeze(0)) for img in x])
    outputs = classifier(x)
    one_hot_labels = to_device(torch.eye(len(outputs[0]))[y])
    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
    j = torch.masked_select(outputs, one_hot_labels.bool())
    if targeted:
        diff = i - j
        stop_attack = diff > kappa
        return (stop_attack * diff), diff
    diff = j - i
    stop_attack = diff > kappa
    return (stop_attack * diff), diff


def PCA_fit_imagenet(path, classifier=None):
    
    
        
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor, Normalize, Resize
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import pdb
    imagenet = ImageFolder(
            path,
            transforms.Compose([
                Resize((256,256)),
                ToTensor(),
            ]))
    imagenet_loader = DataLoader(imagenet, shuffle=True, num_workers=16)
    vector_list = []
    label_list = []
    if classifier:
        
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
    
            def forward(self, x):
                return x
            
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        
        classifier.fc = Identity()
        
        for idx, (image, label) in tqdm(enumerate(imagenet_loader), total=len(imagenet_loader)):
            # if label.item() not in [1, 353, 865, 987, 123]:
            #     continue
            if label.item() > 3:
                continue
            # if idx >= 5000:
            #     break
            cropped_img = center_crop(normalize(to_device(image)), batched=True)
            outputs = classifier(cropped_img).detach().cpu()
            vector_list.append(outputs.view((outputs.shape[0], -1)))
            label_list.append(label)
    else:
        for idx, (image, label) in tqdm(enumerate(imagenet_loader), total=len(imagenet_loader)):
            if label.item() not in [1, 353, 865, 987, 123]:
                continue
            # if idx >= 5000:
            #     break
            vector_list.append(image.view((image.shape[0], -1)))
            label_list.append(label)
    X = torch.cat(vector_list).numpy()
    y = torch.cat(label_list).numpy()
    print(y)
    pca = PCA(n_components=2)
    pca.fit(X)
    mappedX = pca.transform(X)
    
    
    
    plt.scatter(mappedX[:,0], mappedX[:,1], c = y)
    plt.savefig("/home/bar/xb/SAPA/xb/SAPA/attack_results/PCA_visualization__.png")
    
    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c = y)
    plt.savefig("/home/bar/xb/SAPA/xb/SAPA/attack_results/TSNE_visualization__.png")
    
    return

    