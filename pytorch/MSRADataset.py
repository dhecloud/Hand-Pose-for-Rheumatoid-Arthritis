from torch.utils.data import Dataset
import numpy as np

class TrafficSignsDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images)
        self.images = self.images.permute(0, 3, 1, 2)
        self.images = self.images.float()
        print(labels.shape)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def read_depth_from_bin(image_name):
        f = open(image_name, 'rb')
        data = np.fromfile(f, dtype=np.uint32)
        width, height, left, top, right , bottom = data[:6]
        depth = np.zeros((height, width), dtype=np.float32)
        f.seek(4*6)
        data = np.fromfile(f, dtype=np.float32)
        depth[top:bottom, left:right] = np.reshape(data, (bottom-top, right-left))
        return depth
