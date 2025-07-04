﻿# Fire Prediction System

This project is a machine learning-based fire risk prediction system that integrates NASA FIRMS fire data with weather data (from Meteostat) to assess and predict wildfire risk levels in Greece. The system preprocesses, analyzes, and models fire risk using advanced feature engineering and multiple classification algorithms.

## Features

- **Data Integration:** Merges satellite fire detection data with local weather data.
- **Feature Engineering:** Extracts temporal, environmental, and fire-specific features.
- **Fire Risk Categorization:** Uses an improved, multi-criteria risk scoring system.
- **Machine Learning Models:** Trains Random Forest, Logistic Regression, and SVM classifiers.
- **Model Evaluation:** Provides accuracy, classification reports, and confusion matrices.
- **Feature Importance:** Visualizes the most important predictors.
- **Prediction API:** Predicts fire risk for new data samples.
- **Data Analysis:** Analyzes and visualizes fire risk patterns.

## Data Features Explanation

The dataset used in this project combines fire incident data from NASA FIRMS with weather data from Meteostat/Open-Meteo. Below is an explanation of the main features:

### Fire Data Features

- **latitude, longitude:** Geographic coordinates of the fire event.
- **brightness:** Brightness temperature of the fire pixel (Kelvin).
- **scan, track:** Size of the satellite pixel in degrees (scan = width, track = height).
- **acq_date:** Date of fire detection (YYYY-MM-DD).
- **confidence:** Confidence level of the fire detection (e.g., 'n' for nominal).
- **bright_t31:** Brightness temperature of the background (Kelvin).
- **frp:** Fire Radiative Power (MW), an indicator of fire intensity.
- **daynight:** Indicates if the fire was detected during the day ('D') or night ('N').
- **type:** Type of fire event (e.g., 0 = presumed vegetation fire, 2 = active fire, etc.).

### Weather Data Features

- **tavg:** Average daily temperature (°C).
- **tmin:** Minimum daily temperature (°C).
- **tmax:** Maximum daily temperature (°C).
- **prcp:** Daily precipitation (mm).
- **wspd:** Average wind speed.

### Example of Cleaned Data Columns

```
latitude, longitude, brightness, scan, track, acq_date, confidence, bright_t31, frp, daynight, type, tavg, tmin, tmax, prcp, wspd
```

These features are used together to train machine learning models for predicting wildfire


## Project Structure

```
.
├── cleaned.csv                  # Final dataset used for modeling
├── data_preprocessing.py        # Data fetching, merging, and cleaning scripts
├── evia_meteostat_weather.csv   # Raw weather data fetched from Meteostat
├── fire_data.csv                # NASA FIRMS fire data (raw)
├── fire.csv                     # Merged fire and weather data (raw, semicolon-separated)
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── system.py                    # Main ML pipeline and analysis code
├── test.py                      # Meteostat library test script (not important)
├── weather.py                   # (Not used in main pipeline)
```

## Data Sources

- **Fire Data:** NASA FIRMS (fire_data.csv)
- **Weather Data:** Meteostat API (evia_meteostat_weather.csv), Open-Meteo API

## How It Works

1. **Data Preparation:**  
   - Use `data_preprocessing.py` to fetch weather data for each fire event and merge it with fire data.
   - Clean and format the merged data, saving the result as `cleaned.csv`.

2. **Model Training & Evaluation:**  
   - Run `system.py` to:
     - Load and preprocess the data.
     - Engineer features (temporal, environmental, fire-specific).
     - Categorize fire risk using a composite score.
     - Train Random Forest, Logistic Regression, and SVM models.
     - Evaluate models and select the best one.
     - Analyze and visualize fire risk patterns and feature importances.
     - Predict fire risk for new samples.

3. **Prediction:**  
   - The system provides a function to predict fire risk for new input data (see the example in `system.py`).

## Usage

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Prepare Data

- If you downloaded the repository then just move on from step 1 (Installing dependencies) to step 3 (Train and Evaluate models) directly
- Make sure you have `fire.csv` (merged fire and weather data, semicolon-separated).
- Run the cleaning script to produce `cleaned.csv`:

```sh
python data_preprocessing.py
```

### 3. Train and Evaluate Models

```sh
python system.py
```

- The script will output model performance, feature importance, and example predictions.

### 4. Predict Fire Risk for New Data

- Modify the `sample_data` dictionary in `system.py` with your input values and rerun the script, or use the `predict_fire_risk` function programmatically.

## Requirements

See [requirements.txt](requirements.txt).

## Notes

- The system expects the cleaned data to have the following columns:  
  `latitude, longitude, brightness, scan, track, acq_date, confidence, bright_t31, frp, daynight, type, tavg, tmin, tmax, prcp, wspd`
- The main ML pipeline is in [`system.py`](system.py).
- Data fetching and cleaning logic is in [`data_preprocessing.py`](data_preprocessing.py).
- The project is designed for research and educational purposes.

---

## Author

Developed by me Amor Kormadi, 2025 – on AI-powered wildfire prediction in Greece.
