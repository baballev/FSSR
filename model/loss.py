import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGLoss(nn.Module):
    """
    Credits go to the following gist for the content of this file:
    https://https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    """
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).features
        blocks = [vgg16[:4].eval(), vgg16[4:9].eval(), vgg16[9:16].eval(), vgg16[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

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


class Loss:
    @staticmethod
    def get(loss, device):
        if loss == 'VGG':
            return VGGPerceptualLoss().to(device)
        if loss == 'L2':
            return F.mse_loss
        if loss == 'L1':
            return F.l1_loss
        raise NotImplementedError('loss function %s not found' % loss)
