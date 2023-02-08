import torch
from torch.utils.data import Dataset
from models.tree import RstTree


class RstDatasetCoref(Dataset):
    
    def __init__(self, X, y, data_helper, is_train=True):
        self.X = X
        self.y = y
        self.data_helper = data_helper
        self.is_train = is_train

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
            Returns:
                If train: features for that tree (corefs, organizational features, spans, etc.)
                If test: gold tree
        """        
        if self.is_train:
            return self.X[idx], torch.LongTensor(self.y[idx])
        
        doc = self.data_helper.docs[self.X[idx][0]]

        gold_rst = RstTree(doc.filename, doc.filename.replace('.dis', '.merge'))
        gold_rst.build()
        
        return gold_rst
