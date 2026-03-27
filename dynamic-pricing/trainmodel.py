import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("final_dynamic_pricing_dataset.csv")

print("Dataset Loaded:")
print(df.head())

# -------------------------------
# STEP 1: Data Cleaning
# -------------------------------
df = df.dropna()

# -------------------------------
# STEP 2: Encode categorical
# -------------------------------
le = LabelEncoder()
df['season'] = le.fit_transform(df['season'])

# -------------------------------
# STEP 3: Feature Selection
# -------------------------------
features = ['amazon_price','meesho_price','discount_pct','rating',
            'num_reviews','page_views','conversion_rate','demand_intensity','season']

X = df[features]
y = df['price_difference']

# -------------------------------
# STEP 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 5: Train Model (Improved)
# -------------------------------
model = RandomForestRegressor(
    n_estimators=200,      # more trees = better accuracy
    max_depth=10,          # prevent overfitting
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# STEP 6: Evaluate
# -------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 3))

# -------------------------------
# STEP 7: Save EVERYTHING needed
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))
pickle.dump(features, open("features.pkl", "wb"))  # VERY IMPORTANT

print("\nModel, Encoder & Features Saved Successfully!")