# Premier League Match Predictor

A machine learning application that predicts the outcomes of Premier League matches using XGBoost, Random Forest, and Logistic Regression models.

## Features

- Predict match outcomes (Home Win, Draw, Away Win)
- Multiple model support (XGBoost, Random Forest, Logistic Regression)
- Interactive web interface using Streamlit
- Historical match analysis
- Feature importance visualization
- Model performance metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Premier-League-Predictor.git
cd Premier-League-Predictor
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run main.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8502)

3. Select teams and model to get predictions

## Project Structure

- `main.py`: Streamlit web application
- `xgboost_model.py`: XGBoost model training and evaluation
- `random_forest.py`: Random Forest model implementation
- `logistic_regression.py`: Logistic Regression model implementation
- `requirements.txt`: Project dependencies
- `final_dataset.csv`: Training dataset
- `final_testing_dataset.csv`: Testing dataset

## Model Performance

- XGBoost Accuracy: 78.16%
- Random Forest Accuracy: 64.47%
- Logistic Regression Accuracy: 65.86%

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 