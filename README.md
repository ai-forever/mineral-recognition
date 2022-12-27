# Zero-Shot Raw Mineral Visual Recognition and Description

## Preprocessing
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
