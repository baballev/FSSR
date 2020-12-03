import torch.utils.data

class DataLoader(torch.utils.data.DataLoader):
    def __str__(self):
        return str(self.dataset)

    def __repr__(self):
        return repr(self.dataset)
