from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesDataSet(Dataset):
  """
  This is a custom dataset class. It can get more complex than this, but simplified so you can understand what's happening here without
  getting bogged down by the preprocessing
  """
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    _x = np.array(self.X.iloc[index])
    _x = _x.astype(float)
    _y = self.Y.iloc[index]

    return _x, _y




