# Amazon Stock Analysis Dashboard

This interactive dashboard provides comprehensive analysis of Amazon's stock performance from 2000 to 2025. Built with Preswald, it offers various technical indicators, return analytics, and options pricing simulations.

## Features

- **Stock Price Visualization**: Candlestick charts with moving averages
- **Volume Analysis**: Trading volume trends with 20-day moving average
- **Technical Analysis**: Customizable moving averages and Bollinger Bands
- **Return Analytics**: Distribution of daily returns and cumulative return comparisons
- **Volatility Analysis**: Adjustable calculation window for annualized volatility
- **Options Pricing Simulation**: Interactive call/put option payoff diagrams
- **Summary Statistics**: Comprehensive metrics on price changes, returns, and volatility

## Dataset

This application uses historical Amazon stock data from 2000 to 2025, available from:
[Amazon Stock Data on Kaggle](https://www.kaggle.com/datasets/abdulmoiz12/amazon-stock-data-2025)

The dataset contains the following columns:
- Date
- Open price
- High price
- Low price
- Close price
- Adjusted close price
- Trading volume

## Setup Instructions

1. Clone this repository
2. Install Preswald:
   ```
   pip install preswald
   ```
3. Download the dataset from Kaggle (link above)
4. Place the CSV file in the data folder
5. Update the `preswald.toml` file with the correct path to your CSV
6. Run the application:
   ```
   preswald run
   ```

## How to Use

1. **Date Range Selection**: Use the year sliders to select your analysis period
2. **Technical Analysis**: Adjust the moving average window and Bollinger Band parameters
3. **Volatility Analysis**: Modify the calculation window to view different volatility periods
4. **Options Simulation**: Set strike prices and switch between call and put options

## Deployment

You can deploy this dashboard to Preswald's cloud service using:
```
preswald deploy --target structured --github <your-github-username> --api-key <structured-api-key>
```

## Dependencies

- pandas
- numpy
- plotly
- preswald

## Contributing

Feel free to fork this repository and submit pull requests with enhancements or bug fixes.

## License

This project is open source and available under the MIT License.
