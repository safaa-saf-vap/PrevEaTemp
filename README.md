# PrevEaTemp
## Overview
This project is designed to predict rail temperature based on solar effects. It uses a machine learning model and several configuration files to make predictions.

## Project Files

- `TemperaturePrediction.py`: Main script for running the prediction.
- `Config.yaml`: Configuration file containing the settings for the model.
- `PredictionEatempModel.joblib`: Pre-trained model used for prediction.
- `requirements.txt`: List of Python dependencies required to run the project.

## How to Run the Project

### Step 1: Modify the Model Path

In the `TemperaturePrediction.py` script, change the path to point to the location of your `PredictionEatempModel.joblib` file. Find and modify the following line:

```python
model_filename = r'C:\Users\safaa.lahnine\PycharmProjects\Eatemp\PredictionEatempModel.joblib'
