# mineral-recognition

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
