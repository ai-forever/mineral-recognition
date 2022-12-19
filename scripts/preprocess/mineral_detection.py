import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm
import json
import argparse

from object_detection.model import OWLViTModel, preprocess_image


def identical_collate_fn(batch):
    return batch

class ImageDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        pil_image = Image.open(path).convert('RGB')
        img_preproc = Image.fromarray(preprocess_image(pil_image))
        inputs = self.processor(images=[img_preproc], return_tensors="pt")['pixel_values']
        return img_preproc, inputs
        
        
class OWLViTPredictor:
    def __init__(self, owlvit_model, device):
        self.device = device
        self.owlvit_model = owlvit_model
        
    def run(self, df, path_column, num_workers=16, batch_size=32):
        dataset = ImageDataset(df[path_column].values, self.owlvit_model.processor)

        dataloader = DataLoader(
            dataset,
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=identical_collate_fn
        )
        
        labels = {
            'detected_data': [],
        }
        for batch in tqdm(dataloader):
            inputs = {}
            pixel_values = torch.cat([i[1] for i in batch])
            inputs['pixel_values'] = pixel_values
            #
            images = [i[0] for i in batch]
            res = self.owlvit_model._get_boxes_batched(inputs, images)
            labels['detected_data'].extend(res)
            
        labels[path_column] = df[path_column].values
        return pd.DataFrame(labels)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str,
                        help='Path to csv file with image paths.')
    parser.add_argument('--image_path_col', type=str,
                        help='Name of column with image paths.')
    parser.add_argument('--output_csv', type=str,
                        help='Path to output csv.')
    parser.add_argument('--cache_dir', type=str,
                        help='Path to model folder.')
    parser.add_argument('--num_workers', type=int,
                        default=16, help='Number of workers for dataloader.')
    parser.add_argument('--bs', type=int,
                        default=32, help='Batch size.')
    parser.add_argument('--cuda', type=int,
                        help='Id of gpu device.')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    
    device = torch.device(f'cuda:{args.cuda}')
    owl_model = OWLViTModel(
        ["a stone", "a rock", "a mineral", "a gem", "a crystal", "a mineral ore"],
        cache_dir=args.cache_dir,
        device=device
    )
    
    predictor = OWLViTPredictor(owl_model, device)
    result_df = predictor.run(
        df, args.image_path_col, 
        num_workers=args.num_workers,
        batch_size=args.bs
    )
    
    result_df.to_csv(args.output_csv, index=False)