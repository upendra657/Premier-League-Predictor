# Premier League Match Predictor

A machine learning-powered web application that predicts Premier League match outcomes using historical data and XGBoost algorithm.

## Features

- Predict match outcomes between Premier League teams
- Real-time predictions with confidence scores
- Historical data analysis
- Modern, responsive web interface
- RESTful API for predictions

## Tech Stack

### Backend
- Python 3.9
- Flask (Web Framework)
- XGBoost (Machine Learning)
- Pandas & NumPy (Data Processing)
- scikit-learn (Feature Engineering)

### Frontend
- HTML5
- CSS3
- JavaScript (Vanilla)
- Modern UI with responsive design

## Project Structure

```
Premier-League-Predictor/
├── api/
│   ├── app.py              # Flask backend
│   ├── wsgi.py            # WSGI configuration
│   ├── final_dataset.csv  # Historical match data
│   ├── xgboost_model.json # Trained model
│   └── requirements.txt   # Python dependencies
├── frontend/
│   ├── index.html        # Main page
│   ├── styles.css        # Styling
│   └── script.js         # Frontend logic
└── README.md
```

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd api
   pip install -r requirements.txt
   ```

3. Run the Flask server:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Open `index.html` in your browser or serve it using a local server.

## API Endpoints

- `GET /`: Home page
- `POST /predict`: Predict match outcome
  - Request body:
    ```json
    {
      "home_team": "Team Name",
      "away_team": "Team Name",
      "year": 2024
    }
    ```
  - Response:
    ```json
    {
      "predicted_label": "H/NH",
      "confidence": 0.85,
      "actual_result": "H/A/D"  // if available
    }
    ```

## Model Details

- Algorithm: XGBoost Classifier
- Features: Team statistics, historical performance, form
- Output: Home win (H) or Not Home win (NH)
- Performance metrics available in model documentation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Premier League for match data
- XGBoost team for the machine learning library
- Flask team for the web framework 