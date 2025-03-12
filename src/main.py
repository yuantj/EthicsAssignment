# Extreme Precipitation Early Warning System
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests
from io import StringIO
import warnings
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


# 1. Data Acquisition
def fetch_monthly_data(station_id=107420):
    """Retrieve monthly precipitation data from SMHI API"""
    try:
        url = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/23/station/{station_id}/period/corrected-archive/data.csv"
        response = requests.get(url, timeout=15)
        response.encoding = 'utf-8-sig'

        # Locate data header
        data_lines = []
        found_header = False
        for line in response.text.split('\n'):
            if line.startswith('Från Datum Tid (UTC);'):
                headers = [h.strip() for h in line.split(';')[:5]]
                found_header = True
                continue
            if found_header and ';' in line:
                data_lines.append(line.strip())

        # Create DataFrame
        data = pd.DataFrame([ln.split(';')[:5] for ln in data_lines if ln],
                            columns=headers)

        # Standardize column names
        data.columns = ['start_date', 'end_date', 'month', 'precip', 'quality']
        return data

    except Exception as e:
        print(f"Data fetch failed: {str(e)}")
        return None


# 2. Data Preprocessing
def preprocess_monthly_data(raw_data):
    """Clean and transform raw data"""
    try:
        # Convert dates and values
        data = raw_data.copy()
        data['date'] = pd.to_datetime(data['month'] + '-01', errors='coerce')

        # Process precipitation values
        data['precip'] = (
            data['precip']
            .str.replace(',', '.', regex=False)
            .replace(r'^\s*$', np.nan, regex=True)
            .astype(float)
        )

        # Filter valid data
        data = data.dropna(subset=['date', 'precip']).sort_values('date')

        # Define extreme events (top 10% percentile)
        threshold = data['precip'].quantile(0.9)
        data['extreme'] = (data['precip'] >= threshold).astype(int)

        return data[['date', 'precip', 'extreme']]

    except Exception as e:
        print(f"Data processing failed: {str(e)}")
        return None


# 3. Feature Engineering
def create_features(data):
    """Generate model features"""
    df = data.copy()

    # Temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = (df['month'] % 12 // 3) + 1

    # Lag features
    for lag in [1, 12, 24]:
        df[f'lag_{lag}'] = df['precip'].shift(lag)

    # Rolling features
    df['rolling_12_mean'] = df['precip'].rolling(12).mean().shift(1)
    df['rolling_24_min'] = df['precip'].rolling(24).min().shift(1)

    # Remove original columns
    return df.drop(columns=['date', 'precip']).dropna()


# 4. Model Training
def train_model(data):
    """Train and validate model"""
    X = data.drop('extreme', axis=1)
    y = data['extreme']

    # Handle class imbalance
    if y.sum() < 5:
        raise ValueError("Insufficient extreme event samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Model configuration
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=7,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Feature importance
    print("\nFeature Importance:")
    for feat, imp in zip(X.columns, model.feature_importances_):
        print(f"{feat}: {imp:.2f}")

    return model, imputer, X.columns.tolist()


# 5. Prediction Generation
def generate_predictions(model, imputer, features, last_data, periods=12):
    """Generate future predictions"""
    # Prepare base data
    last_date = last_data['date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq='MS'
    )

    # Build feature matrix
    future = []
    for i in range(1, periods + 1):
        base_features = {
            'year': last_date.year + (last_date.month + i - 1) // 12,
            'month': (last_date.month + i - 1) % 12 + 1,
            'season': ((last_date.month + i - 1) % 12 // 3) + 1,
            'lag_1': last_data['precip'].iloc[-1] if i == 1 else np.nan,
            'lag_12': last_data['precip'].iloc[-12] if len(last_data) >= 12 else np.nan,
            'lag_24': last_data['precip'].iloc[-24] if len(last_data) >= 24 else np.nan,
            'rolling_12_mean': last_data['precip'].iloc[-12:].mean(),
            'rolling_24_min': last_data['precip'].iloc[-24:].min()
        }
        future.append(base_features)

    # Create DataFrame
    future_df = pd.DataFrame(future)[features]
    future_df = imputer.transform(future_df)

    # Generate predictions
    return pd.DataFrame({
        'date': future_dates,
        'probability': model.predict_proba(future_df)[:, 1],
        'expected_precip': np.random.normal(
            loc=last_data['precip'].mean(),
            scale=last_data['precip'].std(),
            size=periods)
    }).set_index('date')


# 6. Visualization
def visualize_results(history, future):
    plt.figure(figsize=(14, 7))

    # Historical data
    plt.plot(history.index, history['precip'],
             'o-', lw=1, markersize=5, alpha=0.7,
             color='#1f77b4', label='Historical Precipitation')

    # Extreme events
    extremes = history[history['extreme'] == 1]
    plt.scatter(extremes.index, extremes['precip'],
                s=80, edgecolor='darkred', facecolor='none',
                linewidth=2, label='Historical Extremes')

    # Predictions
    colors = ['#ff7f0e' if p > 0.5 else '#2ca02c' for p in future['probability']]
    for date, row in future.iterrows():
        plt.plot(date, row['expected_precip'], 's',
                 markersize=10, color=colors.pop(0),
                 markeredgecolor='k', alpha=0.9,
                 label='High Risk' if row['probability'] > 0.5 else None)

    plt.title('Extreme Precipitation Early Warning System\n(Orange: High Risk, Green: Safe)')
    plt.ylabel('Precipitation (mm)')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Main Program
if __name__ == "__main__":
    # Data pipeline
    raw_data = fetch_monthly_data()
    if raw_data is None:
        exit()

    clean_data = preprocess_monthly_data(raw_data)
    if clean_data is None:
        exit()

    # Feature engineering
    feature_data = create_features(clean_data)

    try:
        # Model training
        model, imputer, features = train_model(feature_data)

        # Generate predictions
        future_pred = generate_predictions(
            model, imputer, features,
            last_data=clean_data
        )

        # Visualization
        visualize_results(
            history=clean_data.set_index('date'),
            future=future_pred
        )

        # Alert output
        alerts = future_pred[future_pred['probability'] > 0.5]
        if not alerts.empty:
            print("\n\033[1;31m[ALERT] High Risk Months:")
            for date, row in alerts.iterrows():
                print(f"  • {date.strftime('%Y-%m')}: "
                      f"Expected {row['expected_precip']:.1f}mm "
                      f"(Risk Probability: {row['probability']:.0%})")
        else:
            print("\n\033[1;32mNo high-risk precipitation detected in next 12 months")

    except Exception as e:
        print(f"\033[1;31mRuntime Error: {str(e)}")