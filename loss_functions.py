import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# credit goes to github.com/alper111
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).features
        blocks = [vgg16[:4].eval(), vgg16[4:9].eval(), vgg16[9:16].eval(), vgg16[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))

    def forward(self, y_hat, y):
        y_hat = (y_hat-self.mean) / self.std
        y = (y-self.mean) / self.std
        y_hat = self.transform(y_hat, mode='bilinear', size=(224, 224), align_corners=False)
        y = self.transform(y, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        for block in self.blocks:
            y_hat, y = block(y_hat), block(y)
            loss += F.l1_loss(y_hat, y)
        return loss


def perceptionLoss(pretrained_model="vgg16", device=device):
    """
    Possible parameters: "vgg16", "resnet18", "vgg19"
    """
    if pretrained_model == "vgg16":
        m = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features)[:9]).to(device)
    elif pretrained_model == "resnet18":
        tmp_res = torchvision.models.resnet18(pretrained=True)
        m = nn.Sequential(tmp_res.conv1, tmp_res.bn1, tmp_res.relu, tmp_res.maxpool, tmp_res.layer1).to(device)
        del tmp_res
    elif pretrained_model == "vgg19":
        m = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features)[:9]).to(device)
    else:
        raise Exception("Error: Unknown model for perception loss function.")

    def loss(x, y): # Differentiable and the weights of the loss network won't be updated because they were not passed as an argument to the optimizer
        return torch.mean((m(x)-m(y))**2)
    return loss

def reconstructionLoss():
    pass # ToDo


def ultimateLoss(pretrained_model="vgg16", device=device, alpha=0.95, beta=0.05):
    percLoss = perceptionLoss(pretrained_model=pretrained_model, device=device)
    mse = nn.MSELoss()
    def loss(x, y):
        return alpha*mse(x, y) + beta*percLoss(x, y)

    return loss











