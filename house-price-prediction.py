import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -------------------------
# Load dataset
# -------------------------
dataset = pd.read_csv("dataset.csv")

# -------------------------
# Data Cleaning
# -------------------------

dataset["Area"] = pd.to_numeric(dataset["Area"], errors="coerce")

dataset = dataset.dropna(subset=["Area", "Address", "Price", "Price(USD)"])

# حذف outlier برای Area
q_low = dataset["Area"].quantile(0.01)
q_high = dataset["Area"].quantile(0.99)
dataset = dataset[(dataset["Area"] >= q_low) & (dataset["Area"] <= q_high)]

# -------------------------
# Address Encoding (Sorted)
# -------------------------

address_list = sorted(dataset["Address"].unique())

address_to_code = {addr: i for i, addr in enumerate(address_list)}
code_to_address = {i: addr for addr, i in address_to_code.items()}

dataset["Address"] = dataset["Address"].map(address_to_code)

# -------------------------
# Boolean conversion
# -------------------------

for col in ["Parking", "Warehouse", "Elevator"]:
    dataset[col] = dataset[col].astype(int)

# -------------------------
# Features & Targets
# -------------------------

features = ["Area", "Room", "Parking", "Warehouse", "Elevator", "Address"]

X = dataset[features]
y_usd = dataset["Price(USD)"]
y_irr = dataset["Price"]

# -------------------------
# Train models
# -------------------------

X_train, X_test, y_train_usd, y_test_usd = train_test_split(
    X, y_usd, test_size=0.2, random_state=42
)

_, _, y_train_irr, y_test_irr = train_test_split(
    X, y_irr, test_size=0.2, random_state=42
)

model_usd = LinearRegression()
model_irr = LinearRegression()

model_usd.fit(X_train, y_train_usd)
model_irr.fit(X_train, y_train_irr)

# -------------------------
# R2 Score
# -------------------------

r2_usd = r2_score(y_test_usd, model_usd.predict(X_test))
r2_irr = r2_score(y_test_irr, model_irr.predict(X_test))

print("\n📊 Model Performance")
print("R2 USD:", round(r2_usd, 4))
print("R2 IRR:", round(r2_irr, 4))

# -------------------------
# Show addresses (sorted)
# -------------------------

print("\n📍 Available Addresses (Sorted):\n")

for code, addr in code_to_address.items():
    print(f"{code} -> {addr}")

# -------------------------
# User input
# -------------------------

print("\n🏠 Enter house details:\n")

area = float(input("Area (m²): "))
room = int(input("Number of rooms: "))
parking = int(input("Parking (0 or 1): "))
warehouse = int(input("Warehouse (0 or 1): "))
elevator = int(input("Elevator (0 or 1): "))
address_code = int(input("Enter Address Code: "))

# -------------------------
# Prediction (safe format)
# -------------------------

user_df = pd.DataFrame([{
    "Area": area,
    "Room": room,
    "Parking": parking,
    "Warehouse": warehouse,
    "Elevator": elevator,
    "Address": address_code
}])

pred_usd = model_usd.predict(user_df)[0]
pred_irr = model_irr.predict(user_df)[0]

# -------------------------
# Final Output (formatted)
# -------------------------

print("\n💰 Prediction Result")
print("USD:", f"{pred_usd:,.2f}")
print("Toman:", f"{pred_irr:,.2f}")