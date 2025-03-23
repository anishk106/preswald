# Netflix Stock Analysis 📈📊

This project visualizes and analyzes Netflix stock price data from 2002 to 2025. It helps identify trends, patterns, and key events that influenced stock performance over time.

## 📁 Dataset Source

The dataset is sourced from [Kaggle's Netflix Stock Price 2002-2025](https://www.kaggle.com/datasets/samithsachidanandan/netflix-stock-price-2002-2025/data). It contains daily stock prices including open, high, low, close, adjusted close, and trading volume.

## 🚀 Features

- 📊 Interactive Stock Chart: Visualizes Netflix stock price trends with adjustable time periods.
- 📈 Technical Indicators: Displays moving averages, RSI, and MACD for technical analysis.
- 📋 Data Table: Shows raw stock price data with sorting and filtering capabilities.
- ⚡ Powered by Preswald: Simple and fast deployment with `preswald`.

## 🔧 Setup & Running the App

### 💻 Install Dependencies

Ensure you have `preswald` installed:

```
pip install preswald
```

### 🔄 Configure Data Sources

Define your data connections in preswald.toml. Store sensitive information (API keys, passwords) in secrets.toml.

### ▶️ Run the App

Execute the following command to start the app:

```
preswald run hello.py
```

### 🚢 Deploying

To deploy, use:

```
preswald deploy
```

## 📝 Project Structure

```
netflix-stock-analysis/
│
├── data/
│   └── dataset (too big, linked in readme)
│
├── hello.py             # Main application file
├── preswald.toml        # Configuration file
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 📊 Analysis Components

1. Historical Price Analysis
2. Volume Trend Analysis
3. Volatility Measurement
4. Correlation with Market Events
5. Performance Metrics Calculation

## 🔍 Getting Started

1. Clone the repository
2. Install dependencies
3. Run the application
4. Explore the interactive visualizations
