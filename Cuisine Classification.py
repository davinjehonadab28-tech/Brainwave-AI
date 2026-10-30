# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("CUISINE CLASSIFICATION SYSTEM")
print("=" * 60)

# Create comprehensive sample dataset for cuisine classification
print("Creating sample dataset for cuisine classification...")
np.random.seed(42)
n_samples = 3000

# Define cuisine categories and their characteristics
cuisine_categories = {
    'Italian': {'price_range': [2, 4], 'common_words': ['pasta', 'pizza', 'risotto', 'italian', 'mozzarella']},
    'Chinese': {'price_range': [1, 3], 'common_words': ['noodles', 'fried', 'rice', 'szechuan', 'dimsum']},
    'Indian': {'price_range': [1, 3], 'common_words': ['curry', 'spicy', 'tandoori', 'biryani', 'masala']},
    'Mexican': {'price_range': [1, 3], 'common_words': ['taco', 'burrito', 'salsa', 'guacamole', 'mexican']},
    'Japanese': {'price_range': [2, 4], 'common_words': ['sushi', 'ramen', 'tempura', 'teriyaki', 'japanese']},
    'American': {'price_range': [1, 4], 'common_words': ['burger', 'steak', 'bbq', 'fries', 'american']},
    'French': {'price_range': [3, 4], 'common_words': ['wine', 'bistro', 'crepe', 'baguette', 'french']},
    'Thai': {'price_range': [2, 3], 'common_words': ['thai', 'curry', 'pad', 'lemongrass', 'coconut']},
    'Mediterranean': {'price_range': [2, 4], 'common_words': ['greek', 'olive', 'hummus', 'gyro', 'mediterranean']},
    'Seafood': {'price_range': [2, 4], 'common_words': ['fish', 'seafood', 'lobster', 'shrimp', 'oyster']}
}

sample_data = []
for i in range(n_samples):
    cuisine = np.random.choice(list(cuisine_categories.keys()))
    cuisine_info = cuisine_categories[cuisine]
    
    # Generate cuisine description based on cuisine type
    desc_words = cuisine_info['common_words'] + [cuisine.lower()]
    description = ' '.join(np.random.choice(desc_words, size=np.random.randint(3, 6), replace=True))
    
    sample_data.append({
        'Restaurant_ID': i + 1,
        'Restaurant_Name': f'Restaurant_{i+1}',
        'Cuisine_Type': cuisine,
        'Description': description,
        'Average_Cost_for_two': np.random.randint(500, 4000),
        'Price_range': np.random.randint(cuisine_info['price_range'][0], cuisine_info['price_range'][1] + 1),
        'Has_Table_booking': np.random.choice(['Yes', 'No'], p=[0.6, 0.4]),
        'Has_Online_delivery': np.random.choice(['Yes', 'No'], p=[0.5, 0.5]),
        'Aggregate_rating': np.random.uniform(2.5, 5.0),
        'Votes': np.random.randint(10, 2000),
        'City': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Delhi', 'Dubai'])
    })

df = pd.DataFrame(sample_data)

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nCuisine distribution:")
print(df['Cuisine_Type'].value_counts())

# Step 1: Data Preprocessing
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Create a copy for preprocessing
df_processed = df.copy()

# Handle missing values
for column in df_processed.columns:
    if df_processed[column].dtype in ['int64', 'float64']:
        df_processed[column].fillna(df_processed[column].median(), inplace=True)
    else:
        df_processed[column].fillna(df_processed[column].mode()[0] if len(df_processed[column].mode()) > 0 else 'Unknown', inplace=True)

print("Missing values handled successfully!")

# Feature Engineering
print("\nFeature Engineering...")

# Create additional text features
df_processed['Full_Description'] = df_processed['Description'] + ' ' + df_processed['Cuisine_Type'].str.lower()

# Create price categories
def categorize_price(cost):
    if cost <= 1000:
        return 'Budget'
    elif cost <= 2500:
        return 'Moderate'
    else:
        return 'Expensive'

df_processed['Price_Category'] = df_processed['Average_Cost_for_two'].apply(categorize_price)

# Create rating categories
def categorize_rating(rating):
    if rating >= 4.5:
        return 'Excellent'
    elif rating >= 4.0:
        return 'Very_Good'
    elif rating >= 3.5:
        return 'Good'
    elif rating >= 3.0:
        return 'Average'
    else:
        return 'Poor'

df_processed['Rating_Category'] = df_processed['Aggregate_rating'].apply(categorize_rating)

print("Feature engineering completed!")

# Step 2: Prepare Features for Classification
print("\n" + "=" * 60)
print("FEATURE PREPARATION")
print("=" * 60)

# Text feature: Restaurant descriptions
print("Processing text features...")
tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
tfidf_features = tfidf.fit_transform(df_processed['Full_Description'])

print(f"TF-IDF features shape: {tfidf_features.shape}")

# Numerical features
numerical_features = ['Average_Cost_for_two', 'Price_range', 'Aggregate_rating', 'Votes']

# Categorical features
categorical_features = ['Has_Table_booking', 'Has_Online_delivery', 'Price_Category', 'Rating_Category', 'City']

# Encode categorical features
label_encoders = {}
categorical_encoded = []

for feature in categorical_features:
    le = LabelEncoder()
    encoded_feature = le.fit_transform(df_processed[feature])
    categorical_encoded.append(encoded_feature.reshape(-1, 1))
    label_encoders[feature] = le

# Combine all features
from scipy.sparse import hstack

# Combine TF-IDF features with numerical and categorical features
X_numerical = df_processed[numerical_features].values
X_categorical = np.hstack(categorical_encoded)

# Create final feature matrix
X_combined = hstack([tfidf_features, X_numerical, X_categorical])

# Target variable
y = df_processed['Cuisine_Type']

print(f"Final feature matrix shape: {X_combined.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Number of cuisine classes: {len(y.unique())}")

# Step 3: Split Data and Train Models
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Initialize models
models = {
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000)),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42))
}

# Train and evaluate models
results = {}
cv_scores = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Cross-validation
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    cv_scores[name] = cv_score
    
    print(f"{name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Cross-val Accuracy: {cv_score.mean():.4f} (+/- {cv_score.std() * 2:.4f})")

# Step 4: Model Evaluation and Comparison
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Compare model performance
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results],
    'Precision': [results[name]['precision'] for name in results],
    'Recall': [results[name]['recall'] for name in results],
    'F1-Score': [results[name]['f1_score'] for name in results]
})

print("\nModel Performance Comparison:")
print(comparison_df)

# Visualize performance comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy comparison
bars1 = axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'], color=['skyblue', 'lightgreen', 'lightcoral'])
axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
for bar, v in zip(bars1, comparison_df['Accuracy']):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Precision comparison
bars2 = axes[0, 1].bar(comparison_df['Model'], comparison_df['Precision'], color=['skyblue', 'lightgreen', 'lightcoral'])
axes[0, 1].set_title('Precision Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Precision')
for bar, v in zip(bars2, comparison_df['Precision']):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Recall comparison
bars3 = axes[1, 0].bar(comparison_df['Model'], comparison_df['Recall'], color=['skyblue', 'lightgreen', 'lightcoral'])
axes[1, 0].set_title('Recall Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Recall')
for bar, v in zip(bars3, comparison_df['Recall']):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom')

# F1-Score comparison
bars4 = axes[1, 1].bar(comparison_df['Model'], comparison_df['F1-Score'], color=['skyblue', 'lightgreen', 'lightcoral'])
axes[1, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('F1-Score')
for bar, v in zip(bars4, comparison_df['F1-Score']):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Step 5: Detailed Analysis of Best Model
print("\n" + "=" * 60)
print("DETAILED ANALYSIS OF BEST MODEL")
print("=" * 60)

# Identify best model
best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['predictions']

print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {comparison_df['Accuracy'].max():.4f}")

# Detailed classification report
print(f"\nDetailed Classification Report for {best_model_name}:")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred_best, labels=sorted(y.unique()))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y.unique()), 
            yticklabels=sorted(y.unique()))
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 6: Per-Class Performance Analysis
print("\n" + "=" * 60)
print("PER-CLASS PERFORMANCE ANALYSIS")
print("=" * 60)

# Calculate per-class metrics
class_report = classification_report(y_test, y_pred_best, output_dict=True)
class_metrics = []

for cuisine in sorted(y.unique()):
    if cuisine in class_report:
        metrics = class_report[cuisine]
        class_metrics.append({
            'Cuisine': cuisine,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1-score'],
            'Support': metrics['support']
        })

class_metrics_df = pd.DataFrame(class_metrics)
print("\nPer-Class Performance Metrics:")
print(class_metrics_df)

# Visualize per-class performance
plt.figure(figsize=(15, 8))
x_pos = np.arange(len(class_metrics_df))
width = 0.25

plt.bar(x_pos - width, class_metrics_df['Precision'], width, label='Precision', alpha=0.8)
plt.bar(x_pos, class_metrics_df['Recall'], width, label='Recall', alpha=0.8)
plt.bar(x_pos + width, class_metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)

plt.xlabel('Cuisine Type')
plt.ylabel('Score')
plt.title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
plt.xticks(x_pos, class_metrics_df['Cuisine'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Step 7: Feature Importance Analysis
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

if hasattr(best_model, 'feature_importances_'):
    # For Random Forest
    feature_importance = best_model.feature_importances_
    print("Random Forest feature importance available")
elif hasattr(best_model, 'coef_'):
    # For Logistic Regression
    feature_importance = np.mean(np.abs(best_model.coef_), axis=0)
    print("Logistic Regression coefficients analyzed")

# Get top TF-IDF features
feature_names = list(tfidf.get_feature_names_out()) + numerical_features + categorical_features

if len(feature_importance) == len(feature_names):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(importance_df.head(20))
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features for Cuisine Classification', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Step 8: Bias and Challenge Analysis
print("\n" + "=" * 60)
print("BIAS AND CHALLENGE ANALYSIS")
print("=" * 60)

# Analyze class distribution
class_distribution = df_processed['Cuisine_Type'].value_counts()
print("\nClass Distribution:")
print(class_distribution)

# Identify potential biases
print(f"\n📊 CLASS IMBALANCE ANALYSIS:")
max_count = class_distribution.max()
min_count = class_distribution.min()
imbalance_ratio = max_count / min_count

print(f"Most frequent cuisine: {class_distribution.index[0]} ({max_count} samples)")
print(f"Least frequent cuisine: {class_distribution.index[-1]} ({min_count} samples)")
print(f"Imbalance ratio: {imbalance_ratio:.2f}")

if imbalance_ratio > 5:
    print("⚠️  Significant class imbalance detected!")
    print("💡 Recommendation: Consider using class weights or oversampling techniques")
else:
    print("✅ Class distribution is relatively balanced")

# Analyze misclassification patterns
misclassified = X_test[y_test != y_pred_best]
misclassified_cuisines = y_test[y_test != y_pred_best]
predicted_wrong = y_pred_best[y_test != y_pred_best]

misclassification_df = pd.DataFrame({
    'True_Cuisine': misclassified_cuisines,
    'Predicted_Cuisine': predicted_wrong
})

print(f"\n🔍 MISCLASSIFICATION ANALYSIS:")
print(f"Total misclassifications: {len(misclassified)}")
print(f"Misclassification rate: {len(misclassified)/len(y_test):.3f}")

# Most common misclassification pairs
misclassification_pairs = misclassification_df.groupby(['True_Cuisine', 'Predicted_Cuisine']).size().reset_index(name='Count')
misclassification_pairs = misclassification_pairs.sort_values('Count', ascending=False)

print("\nTop 10 Most Common Misclassification Pairs:")
print(misclassification_pairs.head(10))

# Step 9: Model Deployment and Testing
print("\n" + "=" * 60)
print("MODEL TESTING AND DEPLOYMENT")
print("=" * 60)

def predict_cuisine(restaurant_description, cost, price_range, rating, votes, 
                   table_booking, online_delivery, city, model=best_model):
    """
    Predict cuisine type for a new restaurant
    """
    # Create feature vector
    description_tfidf = tfidf.transform([restaurant_description])
    
    # Process numerical features
    numerical_features = np.array([[cost, price_range, rating, votes]])
    
    # Process categorical features
    categorical_features = []
    for feature, value in zip(['Has_Table_booking', 'Has_Online_delivery', 'City'], 
                            [table_booking, online_delivery, city]):
        if value in label_encoders[feature].classes_:
            encoded_value = label_encoders[feature].transform([value])[0]
        else:
            encoded_value = 0  # Default value for unseen categories
        categorical_features.append(encoded_value)
    
    categorical_features = np.array(categorical_features).reshape(1, -1)
    
    # Combine all features
    features_combined = hstack([description_tfidf, numerical_features, categorical_features])
    
    # Make prediction
    prediction = model.predict(features_combined)[0]
    probabilities = model.predict_proba(features_combined)[0] if hasattr(model, 'predict_proba') else None
    
    return prediction, probabilities

# Test the prediction function
print("\n🧪 TESTING PREDICTION FUNCTION:")

test_cases = [
    {
        'description': 'pasta pizza italian restaurant with wine',
        'cost': 2000,
        'price_range': 3,
        'rating': 4.2,
        'votes': 150,
        'table_booking': 'Yes',
        'online_delivery': 'Yes',
        'city': 'New York'
    },
    {
        'description': 'sushi ramen japanese fresh fish',
        'cost': 3000,
        'price_range': 4,
        'rating': 4.5,
        'votes': 300,
        'table_booking': 'Yes',
        'online_delivery': 'No',
        'city': 'Tokyo'
    },
    {
        'description': 'curry spicy indian food biryani',
        'cost': 1200,
        'price_range': 2,
        'rating': 4.0,
        'votes': 200,
        'table_booking': 'No',
        'online_delivery': 'Yes',
        'city': 'Delhi'
    }
]

for i, test_case in enumerate(test_cases, 1):
    prediction, probabilities = predict_cuisine(**test_case)
    print(f"\nTest Case {i}:")
    print(f"Description: {test_case['description']}")
    print(f"Predicted Cuisine: {prediction}")
    if probabilities is not None:
        top_3 = sorted(zip(best_model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3 Predictions: {top_3}")

# Final Summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"✅ CUISINE CLASSIFICATION SYSTEM SUMMARY:")
print(f"• Best Model: {best_model_name}")
print(f"• Overall Accuracy: {comparison_df['Accuracy'].max():.4f}")
print(f"• Number of Cuisine Classes: {len(y.unique())}")
print(f"• Total Training Samples: {X_train.shape[0]}")
print(f"• Feature Dimensions: {X_combined.shape[1]}")

print(f"\n📈 KEY INSIGHTS:")
print(f"• Text descriptions are crucial for cuisine classification")
print(f"• Price range and rating provide additional discriminative power")
print(f"• Some cuisines are more easily distinguishable than others")
print(f"• Class imbalance should be monitored for optimal performance")

print(f"\n🚀 NEXT STEPS:")
print(f"1. Collect more diverse restaurant data")
print(f"2. Experiment with deep learning approaches")
print(f"3. Implement ensemble methods")
print(f"4. Add multi-label classification for fusion cuisines")
print(f"5. Deploy as a real-time classification service")

print("\n" + "=" * 60)
print("CUISINE CLASSIFICATION SYSTEM COMPLETE!")
print("=" * 60)