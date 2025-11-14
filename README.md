# Energy Consumption Analysis Pipeline

## Overview
Python-based analysis pipeline for utility billing data, demonstrating capabilities in:
- Weather normalization using regression analysis
- Anomaly detection in consumption patterns
- Statistical sampling and validation
- Savings opportunity identification
- CSV export functionality for results

## Author
Ricardo Rivera - Data Scientist specializing in energy efficiency and utility analytics

## Features
- **Data Validation**: Comprehensive quality checks with configurable thresholds
- **Weather Normalization**: Regression modeling to separate weather-driven consumption
- **Anomaly Detection**: Z-score based statistical methods to identify unusual patterns
- **Savings Identification**: Weighted scoring algorithm for targeting efficiency programs
- **Error Handling**: Robust exception handling for file operations and data processing
- **Export Functionality**: CSV export for analysis results and opportunities
- **Configurable Parameters**: Adjustable thresholds for IQR, savings estimates, and analysis criteria
- **Automated Reporting**: Summary generation with actionable recommendations

## Technical Stack
- Python 3.8+
- pandas >= 2.2.0
- numpy >= 2.0.0
- scipy >= 1.13.0
- scikit-learn >= 1.5.0

## Configuration
Key parameters can be adjusted in the CONFIG dictionary:
```python
CONFIG = {
    'anomaly_threshold': 2.5,      # Z-score for outliers
    'min_r2': 0.75,                # Min R-squared for weather sensitivity
    'confidence_level': 0.95,       # Statistical confidence
    'baseline_months': 12,          # Baseline period
    'iqr_multiplier': 3,           # IQR multiplier for outlier detection
    'savings_pct': 0.10,           # Estimated savings percentage
    'top_buildings_pct': 0.20      # Top percentage for opportunities
}
```

## Usage
```python
python energy_analysis.py
```

## Input Data Format
The script expects two CSV files:

### Billing Data
- `building_id`: Building identifier
- `date`: Date of billing period
- `kwh`: Energy consumption in kilowatt-hours
- `days_in_period`: Number of days in billing period

### Weather Data
- `date`: Date matching billing periods
- `hdd`: Heating degree days
- `cdd`: Cooling degree days

## Sample Output
The script provides:
- **R² values**: Weather correlation strength (0.341-0.676 typical range)
- **CV-RMSE**: Model accuracy (10-15% indicates good fit)
- **Anomaly detection**: Z-score based outlier identification
- **Priority scoring**: Combined variance and trend analysis
- **Savings estimates**: Conservative 10% from top 20% of buildings
- **CSV export**: `savings_opportunities.csv` with full rankings

## Example Results
```
Buildings analyzed: 3
Weather sensitive: 0
Anomalies detected: 0
Avg model R²: 0.502

Top opportunities identified: 1 buildings
Estimated savings potential: 1,517 kWh/year
```

## Files Generated
- `savings_opportunities.csv`: Ranked list of buildings with savings potential
- Console output with detailed analysis results

## Error Handling
- File not found exceptions
- Missing column validation
- Insufficient data warnings
- Division by zero protection
- Type conversion safety

## Contact
- Email: ricardorivera2008@gmail.com
- LinkedIn: https://www.linkedin.com/in/ricardojulianrivera/
