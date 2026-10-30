# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("RESTAURANT RECOMMENDATION SYSTEM")
print("=" * 60)

# Create sample dataset (replace with your actual dataset)
print("Creating sample dataset...")
np.random.seed(42)
n_samples = 2000

sample_data = {
    'Restaurant ID': range(1, n_samples + 1),
    'Restaurant Name': [f'Restaurant_{i}' for i in range(1, n_samples + 1)],
    'Country Code': np.random.choice([162, 30, 216, 14, 37, 184, 214, 1], n_samples),
    'City': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Delhi', 'Dubai', 'Singapore', 'Mumbai'], n_samples),
    'Cuisines': np.random.choice([
        'Italian, Pizza', 'Chinese, Asian', 'Indian, North Indian', 
        'Mexican, Tex-Mex', 'Japanese, Sushi', 'American, Burger', 
        'French, European', 'Thai, Asian', 'Mediterranean, Greek',
        'Seafood, American', 'BBQ, American', 'Vegetarian, Indian',
        'Korean, Asian', 'Vietnamese, Asian', 'Lebanese, Middle Eastern'
    ], n_samples),
    'Average Cost for two': np.random.randint(300, 5000, n_samples),
    'Currency': np.random.choice(['Dollar($)', 'Indian Rupees(Rs.)', 'Emirati Diram(AED)', 'Botswana Pula(P)'], n_samples),
    'Has Table booking': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
    'Has Online delivery': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
    'Price range': np.random.randint(1, 5, n_samples),
    'Aggregate rating': np.random.uniform(2.0, 5.0, n_samples),
    'Votes': np.random.randint(0, 2000, n_samples)
}

df = pd.DataFrame(sample_data)

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nMissing values:")
print(df.isnull().sum())

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

# Feature Engineering for Recommendation System
print("\nFeature Engineering...")

# Create price categories
def categorize_price(cost):
    if cost <= 1000:
        return 'Budget'
    elif cost <= 2500:
        return 'Moderate'
    else:
        return 'Expensive'

df_processed['Price Category'] = df_processed['Average Cost for two'].apply(categorize_price)

# Create rating categories
def categorize_rating(rating):
    if rating >= 4.5:
        return 'Excellent'
    elif rating >= 4.0:
        return 'Very Good'
    elif rating >= 3.5:
        return 'Good'
    elif rating >= 3.0:
        return 'Average'
    else:
        return 'Poor'

df_processed['Rating Category'] = df_processed['Aggregate rating'].apply(categorize_rating)

# Extract individual cuisines
def get_primary_cuisine(cuisines):
    if ',' in str(cuisines):
        return str(cuisines).split(',')[0].strip()
    return str(cuisines)

df_processed['Primary Cuisine'] = df_processed['Cuisines'].apply(get_primary_cuisine)

print("Feature engineering completed!")
print(f"Available cuisines: {df_processed['Primary Cuisine'].unique()}")
print(f"Price categories: {df_processed['Price Category'].unique()}")

# Step 2: Create Restaurant Feature Matrix
print("\n" + "=" * 60)
print("CREATING RESTAURANT FEATURE MATRIX")
print("=" * 60)

# Select features for recommendation
features_for_rec = [
    'Restaurant ID', 'Restaurant Name', 'City', 'Cuisines', 'Primary Cuisine',
    'Average Cost for two', 'Price Category', 'Price range', 
    'Has Table booking', 'Has Online delivery', 'Aggregate rating', 
    'Rating Category', 'Votes'
]

rec_df = df_processed[features_for_rec].copy()

# Create a combined feature text for content-based filtering
rec_df['Combined_Features'] = (
    rec_df['Primary Cuisine'] + ' ' +
    rec_df['Price Category'] + ' ' +
    rec_df['City'] + ' ' +
    rec_df['Rating Category'] + ' ' +
    rec_df['Has Table booking'] + ' ' +
    rec_df['Has Online delivery']
)

print("Restaurant feature matrix created!")
print(f"Total restaurants in system: {len(rec_df)}")

# Step 3: Implement Content-Based Filtering
print("\n" + "=" * 60)
print("IMPLEMENTING CONTENT-BASED FILTERING")
print("=" * 60)

# TF-IDF Vectorization for text features
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(rec_df['Combined_Features'])

print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Cosine similarity matrix computed!")

# Create mapping between restaurant index and name
indices = pd.Series(rec_df.index, index=rec_df['Restaurant Name']).drop_duplicates()

def get_recommendations(restaurant_name, cosine_sim=cosine_sim, top_n=10):
    """
    Get restaurant recommendations based on content similarity
    """
    try:
        # Get the index of the restaurant
        idx = indices[restaurant_name]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort restaurants by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar restaurants (excluding itself)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get restaurant indices
        restaurant_indices = [i[0] for i in sim_scores]
        
        # Return top N most similar restaurants
        return rec_df[['Restaurant Name', 'Primary Cuisine', 'City', 'Price Category', 
                      'Aggregate rating', 'Votes']].iloc[restaurant_indices]
    
    except KeyError:
        return f"Restaurant '{restaurant_name}' not found in database."

def recommend_by_preferences(cuisine=None, city=None, price_range=None, 
                           min_rating=0, has_booking=None, has_delivery=None, top_n=10):
    """
    Get restaurant recommendations based on user preferences
    """
    filtered_df = rec_df.copy()
    
    # Apply filters
    if cuisine:
        filtered_df = filtered_df[filtered_df['Primary Cuisine'].str.contains(cuisine, case=False, na=False)]
    if city:
        filtered_df = filtered_df[filtered_df['City'].str.contains(city, case=False, na=False)]
    if price_range:
        filtered_df = filtered_df[filtered_df['Price range'] == price_range]
    if min_rating > 0:
        filtered_df = filtered_df[filtered_df['Aggregate rating'] >= min_rating]
    if has_booking:
        filtered_df = filtered_df[filtered_df['Has Table booking'] == has_booking]
    if has_delivery:
        filtered_df = filtered_df[filtered_df['Has Online delivery'] == has_delivery]
    
    # Sort by rating and votes
    filtered_df = filtered_df.sort_values(['Aggregate rating', 'Votes'], ascending=[False, False])
    
    return filtered_df[['Restaurant Name', 'Primary Cuisine', 'City', 'Price Category', 
                       'Price range', 'Aggregate rating', 'Votes', 
                       'Has Table booking', 'Has Online delivery']].head(top_n)

def hybrid_recommendation(user_preferences, top_n=10):
    """
    Hybrid recommendation combining content-based and preference-based filtering
    """
    # First, get preference-based recommendations
    pref_recommendations = recommend_by_preferences(
        cuisine=user_preferences.get('cuisine'),
        city=user_preferences.get('city'),
        price_range=user_preferences.get('price_range'),
        min_rating=user_preferences.get('min_rating', 0),
        has_booking=user_preferences.get('has_booking'),
        has_delivery=user_preferences.get('has_delivery'),
        top_n=top_n * 2  # Get more for filtering
    )
    
    if len(pref_recommendations) == 0:
        return "No restaurants found matching your preferences. Please try different criteria."
    
    # If we have a specific restaurant preference, use content-based filtering
    if user_preferences.get('liked_restaurant'):
        content_recs = get_recommendations(user_preferences['liked_restaurant'], top_n=top_n * 2)
        
        # Combine both approaches and remove duplicates
        if isinstance(content_recs, pd.DataFrame):
            combined = pd.concat([pref_recommendations, content_recs]).drop_duplicates()
            return combined.head(top_n)
    
    return pref_recommendations.head(top_n)

print("Recommendation functions implemented!")

# Step 4: Test the Recommendation System
print("\n" + "=" * 60)
print("TESTING RECOMMENDATION SYSTEM")
print("=" * 60)

# Test Case 1: Content-based recommendations (similar restaurants)
print("\n📋 TEST CASE 1: Content-Based Recommendations")
print("Based on: Restaurant_1")
test_restaurant = "Restaurant_1"
recommendations = get_recommendations(test_restaurant, top_n=5)
print(f"Restaurants similar to '{test_restaurant}':")
print(recommendations)

# Test Case 2: Preference-based recommendations
print("\n📋 TEST CASE 2: Preference-Based Recommendations")
print("Preferences: Italian cuisine, Moderate price, Rating >= 4.0")
user_prefs_1 = {
    'cuisine': 'Italian',
    'price_range': 2,
    'min_rating': 4.0,
    'has_booking': 'Yes'
}
pref_recommendations = recommend_by_preferences(**user_prefs_1, top_n=5)
print("Recommended restaurants:")
print(pref_recommendations)

# Test Case 3: Complex user preferences
print("\n📋 TEST CASE 3: Complex User Preferences")
print("Preferences: Indian cuisine, Budget friendly, With online delivery")
user_prefs_2 = {
    'cuisine': 'Indian',
    'price_range': 1,
    'min_rating': 3.5,
    'has_delivery': 'Yes'
}
pref_recommendations_2 = recommend_by_preferences(**user_prefs_2, top_n=5)
print("Recommended restaurants:")
print(pref_recommendations_2)

# Test Case 4: Hybrid recommendations
print("\n📋 TEST CASE 4: Hybrid Recommendations")
print("Preferences: Liked Restaurant_50, wants similar with Japanese cuisine")
user_prefs_3 = {
    'liked_restaurant': 'Restaurant_50',
    'cuisine': 'Japanese',
    'min_rating': 4.0
}
hybrid_recs = hybrid_recommendation(user_prefs_3, top_n=5)
print("Hybrid recommendations:")
print(hybrid_recs)

# Step 5: Interactive Recommendation System
print("\n" + "=" * 60)
print("INTERACTIVE RECOMMENDATION SYSTEM")
print("=" * 60)

def interactive_recommendation_system():
    """
    Interactive function for users to get personalized recommendations
    """
    print("\n🎯 Welcome to the Restaurant Recommendation System!")
    print("Please provide your preferences (press Enter to skip any criteria):")
    
    preferences = {}
    
    print("\nAvailable cuisines:", list(rec_df['Primary Cuisine'].unique())[:10])
    cuisine = input("Preferred cuisine: ").strip()
    if cuisine:
        preferences['cuisine'] = cuisine
    
    print("\nAvailable cities:", list(rec_df['City'].unique()))
    city = input("Preferred city: ").strip()
    if city:
        preferences['city'] = city
    
    print("\nPrice Range: 1 (Budget) - 4 (Expensive)")
    price_input = input("Preferred price range (1-4): ").strip()
    if price_input and price_input.isdigit():
        preferences['price_range'] = int(price_input)
    
    min_rating_input = input("Minimum rating (0-5): ").strip()
    if min_rating_input:
        preferences['min_rating'] = float(min_rating_input)
    
    table_booking = input("Require table booking? (Yes/No): ").strip()
    if table_booking in ['Yes', 'No']:
        preferences['has_booking'] = table_booking
    
    online_delivery = input("Require online delivery? (Yes/No): ").strip()
    if online_delivery in ['Yes', 'No']:
        preferences['has_delivery'] = online_delivery
    
    liked_restaurant = input("Any restaurant you particularly liked? (Name): ").strip()
    if liked_restaurant:
        preferences['liked_restaurant'] = liked_restaurant
    
    print("\n" + "=" * 50)
    print("YOUR RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = hybrid_recommendation(preferences, top_n=10)
    
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print(f"Found {len(recommendations)} restaurants matching your preferences:")
        print(recommendations)
        
        # Display some statistics
        if len(recommendations) > 0:
            avg_rating = recommendations['Aggregate rating'].mean()
            avg_price_range = recommendations['Price range'].mean()
            print(f"\n📊 Recommendation Stats:")
            print(f"   Average Rating: {avg_rating:.2f}/5")
            print(f"   Average Price Range: {avg_price_range:.1f}/4")

# Run interactive system
interactive_recommendation_system()

# Step 6: System Evaluation and Analysis
print("\n" + "=" * 60)
print("SYSTEM EVALUATION AND ANALYSIS")
print("=" * 60)

# Analyze recommendation diversity
print("\n📈 SYSTEM ANALYSIS")

# Cuisine distribution in recommendations
plt.figure(figsize=(15, 10))

# Plot 1: Cuisine distribution
plt.subplot(2, 2, 1)
cuisine_counts = rec_df['Primary Cuisine'].value_counts().head(10)
plt.bar(cuisine_counts.index, cuisine_counts.values)
plt.title('Top 10 Cuisine Distribution', fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('Number of Restaurants')

# Plot 2: Price range distribution
plt.subplot(2, 2, 2)
price_counts = rec_df['Price Category'].value_counts()
plt.pie(price_counts.values, labels=price_counts.index, autopct='%1.1f%%')
plt.title('Price Category Distribution', fontweight='bold')

# Plot 3: Rating distribution
plt.subplot(2, 2, 3)
rating_bins = [0, 2.5, 3.5, 4.0, 4.5, 5.0]
rating_labels = ['Poor (<2.5)', 'Average (2.5-3.5)', 'Good (3.5-4.0)', 'Very Good (4.0-4.5)', 'Excellent (>4.5)']
rec_df['Rating_Bin'] = pd.cut(rec_df['Aggregate rating'], bins=rating_bins, labels=rating_labels)
rating_dist = rec_df['Rating_Bin'].value_counts()
plt.bar(rating_dist.index, rating_dist.values)
plt.title('Rating Distribution', fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('Number of Restaurants')

# Plot 4: City distribution
plt.subplot(2, 2, 4)
city_counts = rec_df['City'].value_counts().head(8)
plt.bar(city_counts.index, city_counts.values)
plt.title('Top 8 Cities by Restaurant Count', fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('Number of Restaurants')

plt.tight_layout()
plt.show()

# System statistics
print("\n📊 SYSTEM STATISTICS:")
print(f"Total restaurants in database: {len(rec_df)}")
print(f"Number of unique cuisines: {rec_df['Primary Cuisine'].nunique()}")
print(f"Number of cities covered: {rec_df['City'].nunique()}")
print(f"Average restaurant rating: {rec_df['Aggregate rating'].mean():.2f}")
print(f"Restaurants with table booking: {(rec_df['Has Table booking'] == 'Yes').sum()}")
print(f"Restaurants with online delivery: {(rec_df['Has Online delivery'] == 'Yes').sum()}")

# Recommendation quality metrics
print("\n✅ RECOMMENDATION QUALITY METRICS:")
print("1. Content-Based Filtering: Uses cosine similarity on restaurant features")
print("2. Preference-Based Filtering: Matches user criteria with restaurant attributes")
print("3. Hybrid Approach: Combines both methods for personalized recommendations")
print("4. Diversity: Recommends from various cuisines, price ranges, and locations")
print("5. Quality Focus: Prioritizes highly-rated and popular restaurants")

print("\n" + "=" * 60)
print("RECOMMENDATION SYSTEM READY!")
print("=" * 60)
print("\n🎯 System Features:")
print("• Content-based restaurant similarity")
print("• Preference-based filtering")
print("• Hybrid recommendation approach")
print("• Multiple filter criteria (cuisine, price, location, rating, services)")
print("• Interactive user interface")
print("• Comprehensive restaurant database")
print(f"\n🚀 The system is ready to provide personalized restaurant recommendations!")