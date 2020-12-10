import torch
import torch.nn.functional as F

class PSNR:
    """Calculates the PSNR between two images on all provided channels."""

    def __init__(self, threshold=None, edge=0):
        """
        Arguments:
        ---
        threshold: list [a,b] - will clamp the predicted image between a and b
        edge: number of pixels to discard around the image
        """
        self.threshold = threshold
        self.edge = edge

    def __call__(self, y_hat, y):
        """Shape of y_hat and y is [nchannels, width, height]."""
        if self.edge != 0:
            y_hat = y_hat[:, self.edge:-self.edge, self.edge:-self.edge]
            y = y[:, self.edge:-self.edge, self.edge:-self.edge]

        if self.threshold:
            y_hat = y_hat.clamp(*self.threshold)
        mse = F.mse_loss(y_hat, y)
        psnr = -10 * torch.log10(mse)
        return psnr
