# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# Since we don't have the actual CSV file, let's create a sample dataset structure
# In practice, you would replace this with: df = pd.read_csv('Dataset.csv')
print("Creating sample dataset structure...")

# Create sample data based on the structure you provided
np.random.seed(42)
n_samples = 1000

sample_data = {
    'Restaurant ID': range(1, n_samples + 1),
    'Restaurant Name': [f'Restaurant_{i}' for i in range(1, n_samples + 1)],
    'Country Code': np.random.choice([162, 30, 216, 14, 37], n_samples),
    'City': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Delhi', 'Dubai'], n_samples),
    'Cuisines': np.random.choice(['Italian', 'Chinese', 'Indian', 'Mexican', 'Japanese', 'American', 'French'], n_samples),
    'Average Cost for two': np.random.randint(500, 5000, n_samples),
    'Currency': np.random.choice(['Dollar($)', 'Indian Rupees(Rs.)', 'Emirati Diram(AED)'], n_samples),
    'Has Table booking': np.random.choice(['Yes', 'No'], n_samples),
    'Has Online delivery': np.random.choice(['Yes', 'No'], n_samples),
    'Price range': np.random.randint(1, 5, n_samples),
    'Votes': np.random.randint(0, 1000, n_samples),
    'Aggregate rating': np.random.uniform(1.0, 5.0, n_samples)
}

df = pd.DataFrame(sample_data)

# Display basic information about the dataset
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nData types:")
print(df.dtypes)

# Step 1: Data Preprocessing
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Create a copy for preprocessing
df_processed = df.copy()

# Handle missing values
print("Handling missing values...")
for column in df_processed.columns:
    if df_processed[column].dtype in ['int64', 'float64']:
        df_processed[column].fillna(df_processed[column].median(), inplace=True)
    else:
        df_processed[column].fillna(df_processed[column].mode()[0] if len(df_processed[column].mode()) > 0 else 'Unknown', inplace=True)

print(f"Missing values after handling: {df_processed.isnull().sum().sum()}")

# Feature engineering and selection
features_to_use = [
    'Country Code', 'City', 'Cuisines', 'Average Cost for two', 'Currency',
    'Has Table booking', 'Has Online delivery', 'Price range', 'Votes'
]

# Create feature matrix and target variable
X = df_processed[features_to_use].copy()
y = df_processed['Aggregate rating']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encode categorical variables
print("Encoding categorical variables...")
categorical_columns = ['City', 'Cuisines', 'Currency', 'Has Table booking', 'Has Online delivery']

label_encoders = {}
for col in categorical_columns:
    if col in X.columns:
        le = LabelEncoder()
        # Handle any NaN values before encoding
        X[col] = X[col].fillna('Unknown')
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

# Handle 'Has Table booking' and 'Has Online delivery' - convert to binary
if 'Has Table booking' in X.columns:
    X['Has Table booking'] = X['Has Table booking'].map({'Yes': 1, 'No': 0})
if 'Has Online delivery' in X.columns:
    X['Has Online delivery'] = X['Has Online delivery'].map({'Yes': 1, 'No': 0})

# Ensure all values are numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill any remaining NaN values
X = X.fillna(X.median())

print(f"Final feature matrix shape: {X.shape}")
print(f"Feature columns: {X.columns.tolist()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Step 2: Model Training
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"{name} Performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
    except Exception as e:
        print(f"Error training {name}: {e}")
        continue

# Step 3: Model Evaluation and Comparison
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

if results:
    # Compare model performance
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE': [results[name]['mse'] for name in results],
        'RMSE': [results[name]['rmse'] for name in results],
        'MAE': [results[name]['mae'] for name in results],
        'R²': [results[name]['r2'] for name in results]
    })

    print("\nModel Performance Comparison:")
    print(comparison_df)

    # Visualize performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # RMSE comparison
    bars1 = axes[0, 0].bar(comparison_df['Model'], comparison_df['RMSE'], color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 0].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('RMSE')
    for bar, v in zip(bars1, comparison_df['RMSE']):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # R² comparison
    bars2 = axes[0, 1].bar(comparison_df['Model'], comparison_df['R²'], color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].set_title('R² Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('R² Score')
    for bar, v in zip(bars2, comparison_df['R²']):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # Actual vs Predicted for best model
    best_model_name = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
    best_model = results[best_model_name]['model']
    y_pred_best = results[best_model_name]['predictions']

    axes[1, 0].scatter(y_test, y_pred_best, alpha=0.6, color='steelblue')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Ratings')
    axes[1, 0].set_ylabel('Predicted Ratings')
    axes[1, 0].set_title(f'Actual vs Predicted Ratings ({best_model_name})', fontsize=14, fontweight='bold')

    # Residual plot
    residuals = y_test - y_pred_best
    axes[1, 1].scatter(y_pred_best, residuals, alpha=0.6, color='coral')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Ratings')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Step 4: Feature Importance Analysis
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Analyze feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance ({best_model_name}):")
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_importance['feature'], feature_importance['importance'], color='lightseagreen')
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                    ha='left', va='center')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    # For linear regression, show coefficients
    if best_model_name == 'Linear Regression':
        coefficients = pd.DataFrame({
            'feature': X.columns,
            'coefficient': best_model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\nLinear Regression Coefficients:")
        print(coefficients)

    # Step 5: Model Interpretation and Business Insights
    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
    print("=" * 60)

    # Overall performance summary
    best_r2 = comparison_df['R²'].max()
    best_model_final = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']

    print(f"\n🏆 Best Performing Model: {best_model_final}")
    print(f"📊 R² Score: {best_r2:.4f}")

    if best_r2 > 0.7:
        print("✅ The model has strong predictive power")
    elif best_r2 > 0.5:
        print("✅ The model has moderate predictive power")
    else:
        print("⚠️ The model has limited predictive power - consider feature engineering")

    # Key insights about rating drivers
    print(f"\n🔍 KEY FACTORS AFFECTING RESTAURANT RATINGS:")

    if best_model_name in ['Random Forest', 'Decision Tree'] and hasattr(best_model, 'feature_importances_'):
        top_features = feature_importance.head(3)
        for _, row in top_features.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            print(f"• {feature_name}: {importance:.3f} importance")

    # Additional visualizations for insights
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Analyze the relationship between price and rating
    scatter = axes[0].scatter(df_processed['Average Cost for two'], df_processed['Aggregate rating'], 
                             alpha=0.6, c=df_processed['Price range'], cmap='viridis')
    axes[0].set_xlabel('Average Cost for Two')
    axes[0].set_ylabel('Aggregate Rating')
    axes[0].set_title('Price vs Rating Relationship', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[0], label='Price Range')

    # Analyze table booking impact
    if 'Has Table booking' in df_processed.columns:
        booking_impact = df_processed.groupby('Has Table booking')['Aggregate rating'].mean()
        colors = ['lightcoral', 'lightgreen']
        axes[1].bar(['No Table Booking', 'With Table Booking'], booking_impact, color=colors)
        axes[1].set_ylabel('Average Rating')
        axes[1].set_title('Table Booking Impact on Ratings', fontsize=14, fontweight='bold')
        for i, v in enumerate(booking_impact):
            axes[1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Price range impact analysis
    if 'Price range' in df_processed.columns:
        price_impact = df_processed.groupby('Price range')['Aggregate rating'].mean()
        print(f"\n💰 Price Range Impact on Ratings:")
        for price, rating in price_impact.items():
            print(f"   Price Range {price}: {rating:.2f} average rating")

    # Votes impact analysis
    print(f"\n📈 Votes Impact Analysis:")
    print(f"   Average votes for restaurants: {df_processed['Votes'].mean():.1f}")
    print(f"   Correlation between votes and rating: {df_processed['Votes'].corr(df_processed['Aggregate rating']):.3f}")

    print(f"\n💡 RECOMMENDATIONS FOR RESTAURANT OWNERS:")
    print("1. 📋 Focus on maintaining consistent service quality")
    print("2. 🎯 Consider implementing table booking systems")
    print("3. 👥 Encourage customer reviews and engagement (votes)")
    print("4. 💰 Balance pricing with perceived value")
    print("5. 🍽️ Diversify cuisine offerings based on market demand")
    print("6. 📊 Monitor competitor ratings and adapt strategies")
    print("7. 🌟 Invest in online presence and delivery options")

else:
    print("No models were successfully trained.")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

# Display final dataset info
print(f"\nFinal processed dataset info:")
print(f"Total restaurants analyzed: {len(df_processed)}")
print(f"Average rating: {df_processed['Aggregate rating'].mean():.2f}")
print(f"Rating range: {df_processed['Aggregate rating'].min():.2f} - {df_processed['Aggregate rating'].max():.2f}")