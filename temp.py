import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SparseLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Sparse matrix multiplication
        return torch.sparse.mm(input, self.weight.t()) + self.bias

# 사용 예제
input_dim = 600
output_dim = 128
model = SparseLinear(input_dim, output_dim)
sparse_input = torch.sparse.FloatTensor(torch.tensor([[0, 1], [1, 2]]),
                                        torch.tensor([3.0, 4.0]),
                                        torch.Size([10, input_dim]))

output = model(sparse_input)


#########

from torch.utils.data import Dataset, DataLoader

class SparseDataset(Dataset):
    def __init__(self, data, dimension):
        self.data = data
        self.dimension = dimension

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx].split('/')
        indices = torch.tensor(list(map(int, ids)), dtype=torch.long)
        values = torch.ones(len(indices), dtype=torch.float32)
        shape = torch.Size([self.dimension])
        sparse_tensor = torch.sparse.FloatTensor(indices.unsqueeze(0), values, shape)
        return sparse_tensor.to_dense(), torch.tensor(idx)  # 단순 예제로 idx를 타겟으로 사용

# 데이터셋 인스턴스화
dataset = SparseDataset(data, dimension)

# DataLoader 인스턴스화
loader = DataLoader(dataset, batch_size=10000, shuffle=True, num_workers=4)

#########

# 인코더 모델 학습
for epoch in range(10):
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = encoder(inputs)
        loss = criterion(outputs, targets.float())  # 실제 타겟 데이터로 변경 필요
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

########

import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data_indices):
        self.data_indices = data_indices

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        return self.data_indices[idx]

def collate_fn(batch):
    # 배치 단위로 데이터를 변환하는 함수
    return [torch.tensor(indices, dtype=torch.long) for indices in batch]

# 데이터셋 생성
dataset = CustomDataset(data_indices)

# 데이터로더 생성
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# 학습 과정에서 데이터 로드
for batch in dataloader:
    # batch를 이용한 학습 진행
    pass


######

import torch
import torch.nn as nn
import numpy as np

# 데이터 준비 (nested list 형태의 numpy 배열)
data_indices = np.array([[1, 4, 7, 12],
                         [2, 5, 8],
                         [3, 6, 9, 11, 13, 14]])

# 각 샘플의 길이 저장
sample_lengths = [len(indices) for indices in data_indices]

# 데이터를 PyTorch Tensor로 변환
indices_tensor = [torch.tensor(indices, dtype=torch.long) for indices in data_indices]

# 임베딩 레이어 생성
embedding_dim = 20
embedding_layer = nn.EmbeddingBag(num_embeddings=100000, embedding_dim=embedding_dim, mode='sum')

# 임베딩
embedded_vectors = embedding_layer(indices_tensor)

print(embedded_vectors)