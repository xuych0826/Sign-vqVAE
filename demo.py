import torch
from torch.utils.data import Dataset, DataLoader

# 假设您有一个名为CustomDataset的数据集类
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # 在这里可以对数据进行预处理，转换等操作
        return sample

# 假设您有一个名为data的数据列表
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 创建CustomDataset实例
custom_dataset = CustomDataset(data)

# 使用DataLoader加载数据
batch_size = 2
shuffle = True

data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=shuffle)

# 迭代数据加载器以获取数据
for batch in data_loader:
    print(batch)
