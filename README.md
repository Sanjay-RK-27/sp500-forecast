# S&P 500 Forecasting Model

![S&P 500 Forecast](https://img.shields.io/badge/S%26P%20500-Forecast-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-blue)

A machine learning system for predicting S&P 500 price movements using multiple model architectures (LSTM and SVM) with information coefficient (IC) estimation for performance evaluation.

## 📊 Project Overview

This project implements and compares multiple predictive models for forecasting S&P 500 index movements:

- **LSTM Neural Network**: Captures temporal patterns in market data
- **Support Vector Machine**: Identifies critical support/resistance levels
- **Information Coefficient Estimation**: Quantifies prediction quality

The system processes 20 years of historical S&P 500 data to generate forecasts with performance analytics.

## 🔍 Features

- Time series analysis of S&P 500 historical data
- Feature engineering for financial time series
- Multiple model architectures (deep learning and traditional ML)
- Information coefficient (IC) estimation
- Comparative model performance analysis
- Visualization of predictions and actual values

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sanjay-rk-27-sp500-forecast.git
   cd sanjay-rk-27-sp500-forecast
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To train the models:
```bash
python src/train.py
```

To evaluate model performance:
```bash
python src/ic_estimator.py
```

## 📁 Project Structure

```
sanjay-rk-27-sp500-forecast/
├── README.md               # This file
├── requirements.txt        # Project dependencies
├── data/                   # Dataset storage
│   └── sp500_20_years.csv  # Historical S&P 500 data (20 years)
├── models/                 # Saved model artifacts
│   ├── lstm_model.h5       # Trained LSTM model
│   ├── scaler.pkl          # Feature scaling parameters
│   └── svm_model.pkl       # Trained SVM model
├── notebooks/              # Jupyter notebooks for exploration
│   └── 01-initial-ic.ipynb # Initial information coefficient analysis
├── plots/                  # Generated visualizations
└── src/                    # Source code
    ├── data_utils.py       # Data loading and preprocessing utilities
    ├── features.py         # Feature engineering
    ├── ic_estimator.py     # Information coefficient calculation
    ├── model_lstm.py       # LSTM model implementation
    ├── model_svm.py        # SVM model implementation
    └── train.py            # Model training script
```

## 📈 Model Performance

The models are evaluated using the Information Coefficient (IC), which measures the correlation between predicted returns and actual returns. Higher IC values indicate better predictive performance.

| Model | Mean IC | IC Volatility | IR (IC/IC Vol) |
|-------|---------|---------------|----------------|
| LSTM  | 0.21    | 0.09          | 2.33           |
| SVM   | 0.17    | 0.12          | 1.42           |

## 🧪 Methodology

1. **Data Preprocessing**:
   - Removing outliers and handling missing values
   - Feature normalization using StandardScaler
   - Time series splitting for training and testing

2. **Feature Engineering**:
   - Technical indicators (Moving averages, RSI, MACD)
   - Volatility measurements
   - Lagged features for sequential pattern recognition

3. **Model Training**:
   - LSTM: Designed for sequence learning and temporal patterns
   - SVM: Captures non-linear relationships with RBF kernel

4. **Performance Evaluation**:
   - Information Coefficient (IC) calculation
   - Visualizations of predicted vs. actual values
   - Statistical significance testing

## 🛠️ Technologies

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning implementation
- **scikit-learn**: Traditional machine learning algorithms
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please contact:
- Email: sanjayraja27@icloud.com
- GitHub: [@Sanjay-RK-27](https://github.com/Sanjay-RK-27)

## 🙏 Acknowledgements

- S&P 500 historical data providers
- The open-source community for machine learning libraries
- Financial forecasting research papers that inspired this project
