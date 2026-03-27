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

# Drop missing values
df = df.dropna()

# Encode season (categorical)
le = LabelEncoder()
df['season'] = le.fit_transform(df['season'])

# Features
X = df[['amazon_price','meesho_price','discount_pct','rating',
        'num_reviews','page_views','conversion_rate','demand_intensity','season']]

# Target
y = df['price_difference']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("Model & Encoder Saved!")