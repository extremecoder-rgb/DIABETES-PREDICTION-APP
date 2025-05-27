# Diabetes Risk Assessment Tool

This is a machine learning-powered web application that helps users assess their risk of developing diabetes based on various health indicators and lifestyle factors.

## Features

- Interactive questionnaire to collect health and lifestyle data
- Machine learning-based risk assessment
- Real-time risk calculation
- User-friendly interface
- Detailed health recommendations based on risk level

## Installation

1. Clone the repository:
```bash
git clone https://github.com/extremecoder-rgb/DIABETES-PREDICTION-APP.git
cd DIABETES-PREDICTION-APP
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python models/model_training.py
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app/main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Fill out the questionnaire with your health and lifestyle information

4. Receive your personalized diabetes risk assessment

## Project Structure

```
DIABETES-PREDICTION-APP/
├── app/
│   └── main.py              # Streamlit application
├── models/
│   ├── model_training.py    # Model training script
│   └── *.pkl               # Trained model files
├── data/
│   └── diabetes_binary_health_indicators_BRFSS2015.csv
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Model Information

The application uses multiple machine learning models:
- Logistic Regression
- Random Forest
- Gradient Boosting

The models are trained on the BRFSS 2015 dataset and predict the probability of diabetes based on various health indicators.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- BRFSS 2015 dataset
- Streamlit for the web application framework
- scikit-learn for machine learning capabilities 