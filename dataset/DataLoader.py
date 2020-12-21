import torch.utils.data

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoader, self).__init__(pin_memory=True, *args, **kwargs)
    
    def __str__(self):
        return str(self.dataset)

    def __repr__(self):
        return repr(self.dataset)
