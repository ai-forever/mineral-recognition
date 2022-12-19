import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm
import json
import argparse

from CRAFT.model import CRAFTModel, boxes_area


def identical_collate_fn(batch):
    return batch

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        return np.array(img)
        
        
class CRAFTPredictor:
    def __init__(self, craft_model, device):
        self.device = device
        self.craft_model = craft_model
        
    def run(self, df, path_column, num_workers=16):
        dataset = ImageDataset(df[path_column].values)
    
        dataloader = DataLoader(
            dataset,
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=1,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=identical_collate_fn
        )
        
        labels = {
            'text_boxes': [],
            'text_area': [],
            'num_boxes': []
        }
        for img in tqdm(dataloader):
            img = img[0]
            with torch.no_grad():
                bboxes = self.craft_model.get_boxes(img)

            labels['text_boxes'].append(json.dumps(bboxes))
            labels['text_area'].append(boxes_area(img, bboxes))
            labels['num_boxes'].append(len(bboxes))
            
        predicted_df = pd.DataFrame(labels)
        predicted_df[path_column] = df[path_column].values
        
        return predicted_df
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str,
                        help='Path to csv file with image paths.')
    parser.add_argument('--image_path_col', type=str,
                        help='Name of column with image paths.')
    parser.add_argument('--output_csv', type=str,
                        help='Path to output csv.')
    parser.add_argument('--model_path', type=str,
                        help='Path to CRAFT model.')
    parser.add_argument('--refiner_path', type=str,
                        help='Path to refiner model.')
    parser.add_argument('--num_workers', type=int,
                        default=16, help='Number of workers for dataloader.')
    parser.add_argument('--cuda', type=int,
                        help='Id of gpu device.')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    
    device = torch.device(f'cuda:{args.cuda}')
    model = CRAFTModel(
        model_path=args.model_path,
        refiner_path=args.refiner_path,
        device=device
    )
    
    predictor = CRAFTPredictor(model, device)
    result_df = predictor.run(df, args.image_path_col, num_workers=args.num_workers)
    
    result_df.to_csv(args.output_csv, index=False)