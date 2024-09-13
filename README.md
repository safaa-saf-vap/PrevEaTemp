# PrevEaTemp
## Overview
This project is designed to predict rail temperature based on weather and solar effects.

## Project Files

- `TemperaturePrediction.py`: Main script for running the model.
- `Config.yaml`: Configuration file containing the settings for the model.
- `PredictionEatempModel.joblib`: Pre-trained model used for prediction.
- `requirements.txt`: List of Python dependencies required to run the project.

## How to Run the Project
use Python version 3.9 or later
### Step 1: Modify the Model Path

In the `TemperaturePrediction.py` script, change the path to point to the location of your `PredictionEatempModel.joblib` file. Find and modify the following line:

```python
model_filename = r'C:\Users\safaa.lahnine\PycharmProjects\Eatemp\PredictionEatempModel.joblib'
```

### Step 2:  install the required dependencies
```bash
pip install -r requirements.txt
pip install openmeteo_requests requests_cache pandas retry_requests PyYaml joblib xgboost
```
### Step 3: Run the code
```bash
python TemperaturePrediction.py
```

