# AIPI540 A2

## install dependencies

```bash
pip install -r requirements.txt
```


## Data Preprocessing

By running the following command, the data will be preprocessed and saved to `models/data/processed_data.csv`.

```bash
python scripts/preprocess_data.py
```

To check the quality of the processed data, run the following command:

```bash
python scripts/check_label_quality.py
```