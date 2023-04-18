from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image

from dataclasses import dataclass
from dataingestion import DataIngest
from src.utils import TwoCropTransform

@dataclass
class DataIngestConfig:
    config = DataIngest()
    train_idx, test_idx, valid_idx = config.initiate_data_ingestion()
    
    DF_train_data = config.idx_to_path('Deepfakes', train_idx)
    DF_valid_data = config.idx_to_path('Deepfakes', valid_idx)
    DF_test_data = config.idx_to_path('Deepfakes', test_idx)

    F2F_train_data = config.idx_to_path('Face2Face', train_idx)
    F2F_valid_data = config.idx_to_path('Face2Face', valid_idx)
    F2F_test_data = config.idx_to_path('Face2Face', test_idx)

    FS_train_data = config.idx_to_path('FaceSwap', train_idx)
    FS_valid_data = config.idx_to_path('FaceSwap', valid_idx)
    FS_test_data = config.idx_to_path('FaceSwap', test_idx)

    NT_train_data = config.idx_to_path('NeuralTextures', train_idx)
    NT_valid_data = config.idx_to_path('NeuralTextures', valid_idx)
    NT_test_data = config.idx_to_path('NeuralTextures', test_idx)

    OR_train_data = config.idx_to_path('Original', train_idx)
    OR_valid_data = config.idx_to_path('Original', valid_idx)
    OR_test_data = config.idx_to_path('Original', test_idx)




class MyDataset(Dataset):
    def __init__(self, data_frame):
        
        self.data = data_frame
        
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
        self._single_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        self.transform = TwoCropTransform(self._single_transform)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path = self.data.loc[idx, 'path']
        label = self.data.loc[idx, 'labels']
        # RGB
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label

