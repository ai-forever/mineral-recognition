# Dataset
[The dataset is available online](https://disk.yandex.ru/d/KapicF_MEysifg)
* `mineral_images.zip` contains mineral images
* `mineral_full.csv` contains all mineral samples descriptions
* `minerals_10.csv`, `minerals_98.csv` and `minerals_360.csv` contain mineral samples splits
* `minerals_size.csv` and `segm.tar` contain auxiliary information for some samples



# Zero-Shot Raw Mineral Visual Recognition and Description

This repository provides code for mineral recognition experiments. We explore zero-shot problems on raw mineral samples. The dataset and the paper will be shared later. 

## Preprocessing
During data preprocessing, we obtain zero-shot detection to locate text tables, reference cubes and minerals themselves. 

Run text detection
```bash
python scripts/preprocess/text_detection.py --input_csv=data/data.csv --image_path_col=image_path --output_csv=data/text_detection_res.csv --model_path=weights/craft_mlt_25k.pth --refiner_path=weights/craft_refiner_CTW1500.pth --cuda=0
```
Result will be saved in `output_csv`

---

Run mineral (object) detection
```bash
python scripts/preprocess/mineral_detection.py --input_csv=data/data.csv --image_path_col=image_path --output_csv=data/mineral_detection_res.csv --cache_dir=weights/ --cuda=0
```
Result will be saved in `output_csv`


## Size estimator
Run predict size mineral
```bash
python scripts/predict/size_estimator.py --input_csv=data/data.csv  --output_csv=data/predict_size.csv
```
Result will be saved in `predict_size.csv`


---

Run eval size mineral
```bash
python scripts/eval/eval_size_estimator.py --predict_csv=data/predict_size.csv --ground_true_csv=data/true_size.csv
```


## Zero-shot classification
For zero-shot classification, we use [CLIP](https://github.com/openai/CLIP).



## Zero-shot segmentation
For zero-shot segmentation, we apply [GradCAM](https://github.com/jacobgil/pytorch-grad-cam) techniques to a pre-trained classifier.




