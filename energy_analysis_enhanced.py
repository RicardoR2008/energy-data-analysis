"""
Energy Consumption Analysis Pipeline
Author: Ricardo Rivera
Purpose: Analyze utility billing data to identify savings opportunities
         using regression analysis and anomaly detection
Date: November 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration settings
CONFIG = {
    'anomaly_threshold': 2.5,  # Z-score for outliers
    'min_r2': 0.75,  # Min R-squared for weather sensitivity
    'confidence_level': 0.95,
    'baseline_months': 12,
    'iqr_multiplier': 3,  # Made configurable
    'savings_pct': 0.10,  # 10% savings estimate
    'top_buildings_pct': 0.20  # Top 20% for opportunities
}

class EnergyAnalyzer:
    """
    Main analysis class for building energy consumption.
    Handles weather normalization and anomaly detection.
    """
    
    def __init__(self, config=CONFIG):
        """Set up analyzer with config params."""
        self.config = config
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_validate_data(self, billing_path, weather_path):
        """
        Load billing and weather data, validate, and merge.
        
        Args:
            billing_path: Path to billing CSV
            weather_path: Path to weather CSV
            
        Returns:
            DataFrame with merged data
        """
        print("Loading data...")
        
        try:
            # Read CSVs
            billing = pd.read_csv(billing_path)
            weather = pd.read_csv(weather_path)
        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
        # Check we have the columns we need
        required_billing = ['building_id', 'date', 'kwh', 'days_in_period']
        required_weather = ['date', 'hdd', 'cdd']  # heating/cooling degree days
        
        self._validate_columns(billing, required_billing, 'billing')
        self._validate_columns(weather, required_weather, 'weather')
        
        # Parse dates
        billing['date'] = pd.to_datetime(billing['date'])
        weather['date'] = pd.to_datetime(weather['date'])
        
        # Get daily usage rate
        billing['kwh_per_day'] = billing['kwh'] / billing['days_in_period']
        
        # Join on date
        data = pd.merge(billing, weather, on='date', how='inner')
        
        # Run quality checks
        self._perform_quality_checks(data)
        
        print(f"Loaded {len(data)} records for {data['building_id'].nunique()} buildings")
        
        return data
    
    def _validate_columns(self, df, required_cols, dataset_name):
        """Make sure we have all needed columns."""
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {dataset_name}: {missing}")
    
    def _perform_quality_checks(self, data):
        """Check data quality and flag any issues."""
        issues = []
        
        # Look for missing values
        null_counts = data[['kwh', 'hdd', 'cdd']].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts.to_dict()}")
        
        # Flag negative consumption (shouldn't happen)
        negative = data['kwh'] < 0
        if negative.any():
            issues.append(f"Negative consumption: {negative.sum()} records")
            data.loc[negative, 'kwh'] = np.nan
        
        # Find extreme outliers with IQR (now configurable)
        Q1 = data['kwh_per_day'].quantile(0.25)
        Q3 = data['kwh_per_day'].quantile(0.75)
        IQR = Q3 - Q1
        multiplier = self.config.get('iqr_multiplier', 3)
        outliers = ((data['kwh_per_day'] < (Q1 - multiplier * IQR)) | 
                   (data['kwh_per_day'] > (Q3 + multiplier * IQR)))
        
        if outliers.any():
            issues.append(f"Extreme outliers: {outliers.sum()} records")
        
        if issues:
            print("Data Quality Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Data quality checks passed")
            
        return data
    
    def perform_weather_normalization(self, data, building_id):
        """
        Run weather normalization regression for a building.
        
        Args:
            data: DataFrame with energy/weather data
            building_id: Building to analyze
            
        Returns:
            Dict with regression results and anomalies
        """
        print(f"\nPerforming weather normalization for building {building_id}...")
        
        # Get this building's data
        bldg_data = data[data['building_id'] == building_id].copy()
        
        if len(bldg_data) < 3:  # Need minimum data points
            print(f"  Warning: Insufficient data for {building_id}")
            return None
        
        # Set up features and target
        X = bldg_data[['hdd', 'cdd']].values
        y = bldg_data['kwh_per_day'].values
        
        # Drop any NaNs
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 3:  # Check again after cleaning
            print(f"  Warning: Insufficient clean data for {building_id}")
            return None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit regression
        self.model.fit(X_scaled, y_clean)
        y_pred = self.model.predict(X_scaled)
        
        # Get model stats
        r2 = self.model.score(X_scaled, y_clean)
        rmse = np.sqrt(np.mean((y_clean - y_pred) ** 2))
        cv_rmse = rmse / np.mean(y_clean) * 100  # coefficient of variation
        
        # Find anomalies using residuals
        residuals = y_clean - y_pred
        if len(residuals) > 1:  # Need multiple points for z-score
            z_scores = np.abs(stats.zscore(residuals))
            anomalies = z_scores > self.config['anomaly_threshold']
        else:
            anomalies = np.array([False] * len(residuals))
        
        results = {
            'building_id': building_id,
            'r2': r2,
            'rmse': rmse,
            'cv_rmse': cv_rmse,
            'baseline_consumption': np.mean(y_clean),
            'weather_sensitive': r2 > self.config['min_r2'],
            'anomaly_count': int(np.sum(anomalies)),  # Convert to int
            'anomaly_dates': bldg_data.loc[mask, 'date'].iloc[anomalies].tolist() if anomalies.any() else [],
            'coefficients': {
                'hdd': float(self.model.coef_[0]),  # Convert to float
                'cdd': float(self.model.coef_[1]),
                'intercept': float(self.model.intercept_)
            }
        }
        
        print(f"  R²: {r2:.3f}")
        print(f"  CV-RMSE: {cv_rmse:.1f}%")
        print(f"  Anomalies detected: {results['anomaly_count']}")
        
        return results
    
    def identify_savings_opportunities(self, data):
        """
        Find buildings with best savings potential.
        
        Args:
            data: Building energy DataFrame
            
        Returns:
            DataFrame ranked by opportunity
        """
        print("\nIdentifying savings opportunities...")
        
        opportunities = []
        
        for building_id in data['building_id'].unique():
            bldg_data = data[data['building_id'] == building_id]
            
            # Get consumption metrics
            consumption = bldg_data['kwh'].sum()
            
            # Handle variance calculation
            if bldg_data['kwh_per_day'].std() > 0:
                variance = bldg_data['kwh_per_day'].std() / bldg_data['kwh_per_day'].mean()
            else:
                variance = 0
            
            # Look for upward trend
            x = np.arange(len(bldg_data))
            y = bldg_data['kwh_per_day'].values
            if len(x) > 2:
                slope, intercept = np.polyfit(x, y, 1)
                trend = slope / np.mean(y) * 100  # % change per period
            else:
                trend = 0
            
            opportunities.append({
                'building_id': building_id,
                'annual_consumption': consumption * (365 / bldg_data['days_in_period'].sum()),
                'variance_coefficient': variance,
                'trend_percent': trend,
                'priority_score': variance * 0.4 + abs(trend) * 0.6  # weighted combo
            })
        
        opportunities_df = pd.DataFrame(opportunities)
        opportunities_df = opportunities_df.sort_values('priority_score', ascending=False)
        
        # Estimate savings using config values
        top_pct = self.config.get('top_buildings_pct', 0.20)
        savings_pct = self.config.get('savings_pct', 0.10)
        top_n = max(1, int(len(opportunities_df) * top_pct))  # At least 1 building
        potential_savings = opportunities_df.head(top_n)['annual_consumption'].sum() * savings_pct
        
        print(f"\nTop opportunities identified: {top_n} buildings")
        print(f"Estimated savings potential: {potential_savings:,.0f} kWh/year")
        
        return opportunities_df
    
    def generate_summary_report(self, data, results_list):
        """
        Create final summary with stats and recommendations.
        
        Args:
            data: Original DataFrame
            results_list: List of weather norm results
            
        Returns:
            Dict with summary report
        """
        print("\nGenerating summary report...")
        
        # Filter out None results
        valid_results = [r for r in results_list if r is not None]
        
        if not valid_results:
            print("Warning: No valid results to report")
            return None
        
        weather_sensitive = sum(1 for r in valid_results if r['weather_sensitive'])
        total_anomalies = sum(r['anomaly_count'] for r in valid_results)
        avg_r2 = np.mean([r['r2'] for r in valid_results])
        
        report = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'buildings_analyzed': len(valid_results),
            'date_range': {
                'start': data['date'].min().strftime('%Y-%m-%d'),
                'end': data['date'].max().strftime('%Y-%m-%d')
            },
            'summary_statistics': {
                'total_consumption_kwh': float(data['kwh'].sum()),  # Convert to float
                'avg_daily_consumption': float(data['kwh_per_day'].mean()),
                'weather_sensitive_buildings': weather_sensitive,
                'weather_sensitivity_pct': weather_sensitive / len(valid_results) * 100 if valid_results else 0,
                'avg_model_r2': float(avg_r2),
                'total_anomalies': total_anomalies
            },
            'recommendations': []
        }
        
        # Build recommendations based on what we found
        if avg_r2 > 0.7:
            report['recommendations'].append(
                "Strong weather correlation detected. Consider weather-based targeting for DSM programs."
            )
        
        if total_anomalies > len(valid_results) * 2:
            report['recommendations'].append(
                "High anomaly rate detected. Investigate operational issues or metering problems."
            )
        
        if len(valid_results) > 0 and weather_sensitive / len(valid_results) > 0.6:
            report['recommendations'].append(
                "Majority of buildings are weather-sensitive. HVAC optimization programs recommended."
            )
            
        # Add recommendation if no recommendations generated
        if not report['recommendations']:
            report['recommendations'].append(
                "Continue monitoring. Consider detailed energy audits for high-consumption buildings."
            )
        
        return report

def export_results(results_df, output_path='energy_analysis_results.csv'):
    """
    Export results to CSV file.
    
    Args:
        results_df: DataFrame with results
        output_path: Path for output file
    """
    try:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults exported to {output_path}")
    except Exception as e:
        print(f"Error exporting results: {e}")

# Demo function
def main():
    """
    Run the analysis pipeline demo.
    """
    print("=" * 60)
    print("ENERGY CONSUMPTION ANALYSIS PIPELINE")
    print("Demonstrating capabilities for utility program evaluation")
    print("=" * 60)
    
    # Set up analyzer
    analyzer = EnergyAnalyzer()
    
    # Generate sample data for demo
    sample_data = create_sample_data()
    
    # Run weather normalization on first 3 buildings
    results_list = []
    for building_id in sample_data['building_id'].unique()[:3]:
        results = analyzer.perform_weather_normalization(sample_data, building_id)
        if results:  # Only add valid results
            results_list.append(results)
    
    # Find best opportunities
    opportunities = analyzer.identify_savings_opportunities(sample_data)
    print("\nTop 5 Buildings by Savings Potential:")
    print(opportunities[['building_id', 'annual_consumption', 'priority_score']].head())
    
    # Export opportunities to CSV
    export_results(opportunities, 'savings_opportunities.csv')
    
    # Generate final report
    report = analyzer.generate_summary_report(sample_data, results_list)
    
    if report:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Buildings analyzed: {report['buildings_analyzed']}")
        print(f"Weather sensitive: {report['summary_statistics']['weather_sensitive_buildings']}")
        print(f"Anomalies detected: {report['summary_statistics']['total_anomalies']}")
        print(f"Avg model R²: {report['summary_statistics']['avg_model_r2']:.3f}")
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return report

def create_sample_data():
    """Generate fake data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='MS')
    buildings = ['BLDG_001', 'BLDG_002', 'BLDG_003', 'BLDG_004', 'BLDG_005']
    
    data = []
    for building in buildings:
        for date in dates:
            # Simulate seasonal patterns
            month = date.month
            base_load = np.random.normal(1000, 100)
            
            # Summer months - higher cooling
            if month in [6, 7, 8]:
                kwh = base_load + np.random.normal(500, 50)
                cdd = np.random.uniform(10, 25)
                hdd = 0
            # Winter months - higher heating  
            elif month in [12, 1, 2]:
                kwh = base_load + np.random.normal(400, 40)
                hdd = np.random.uniform(15, 30)
                cdd = 0
            # Spring/Fall - mild
            else:
                kwh = base_load
                hdd = np.random.uniform(0, 10)
                cdd = np.random.uniform(0, 10)
            
            # Calculate daily rate
            days_in_period = 30
            kwh_value = max(0, kwh)  # no negative values
            
            data.append({
                'building_id': building,
                'date': date,
                'kwh': kwh_value,
                'days_in_period': days_in_period,
                'kwh_per_day': kwh_value / days_in_period,
                'hdd': hdd,
                'cdd': cdd
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Run it
    results = main()
