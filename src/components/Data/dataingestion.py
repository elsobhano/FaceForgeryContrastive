import os
import sys
from src.exception import CustomExeption
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
import json
from typing import Literal

@dataclass
class DataIngestConfig:
    
    train_path: str= os.path.join('artifacts','train_idx.txt')
    test_path: str= os.path.join('artifacts','test_idx.txt')
    valid_path: str= os.path.join('artifacts','valid_idx.txt')    

class DataIngest:
    def __init__(self) -> None:
        self.config = DataIngestConfig()

    def initiate_data_ingestion(self):
        
        logging.info('Entered the data ingestion method')
        
        try:

            with open(self.config.train_path, 'r') as file:
                self.train_idx = json.load(file)
        
            with open(self.config.test_path, 'r') as file:
                self.test_idx = json.load(file)
        
            with open(self.config.valid_path, 'r') as file:
                self.valid_idx = json.load(file)

            logging.info('Loading train, test, valid indexs is completed')

            return(
                self.train_idx,
                self.test_idx,
                self.valid_idx
            )
        except Exception as e:
            raise CustomExeption(e, sys)
    
    MODE = Literal['Original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    def idx_to_path(self, cat: MODE, indexes: list) -> pd.DataFrame:
        root = 'notebook/dataset/c23'
        data_frame = {'path':[]}
        path = root + '/{}'.format(cat)
        for idx in indexes:
            folder_path = path + '/{}'.format(idx)
            if os.path.exists(folder_path) == True:
                for file_name in os.listdir(folder_path):
                    file_path = folder_path + '/{}'.format(file_name)
                    data_frame['path'].append(file_path)

        if cat == 'Original':
            labels = [1]*len(data_frame['path'])
        else:
            labels = [0]*len(data_frame['path'])

        data_frame['labels'] = labels
        data_frame = pd.DataFrame(data_frame)
        
        return data_frame

if __name__=="__main__":
    obj = DataIngest()
    obj.initiate_data_ingestion()