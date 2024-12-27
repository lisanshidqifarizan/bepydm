from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mangum import Mangum

# FastAPI app initialization
app = FastAPI()

# Tambahkan daftar origin yang diizinkan
origins = [
    "http://localhost:3000",  # Frontend local development
    "https://fejsdm.vercel.app"  # Tambahkan domain frontend produksi Anda (jika ada)
]

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Ganti "*" dengan daftar origin
    allow_credentials=True,
    allow_methods=["*"],  # Ijinkan semua metode (GET, POST, dll.)
    allow_headers=["*"],  # Ijinkan semua header
)

# Load dataset from topuniversities.csv
try:
    data = pd.read_csv('topuniversities.csv')  # Ensure this file exists in your project root
    data = data[['University Name', 'Overall Score', 'Citations per Paper', 'Papers per Faculty', 
                 'Academic Reputation', 'Faculty Student Ratio']]  # Select relevant columns
    data = data.dropna()  # Remove rows with missing values
except FileNotFoundError:
    raise Exception("The file 'topuniversities.csv' was not found. Ensure it is in the project directory.")

# Process data
data['Success'] = (data['Overall Score'] >= 50).astype(int)
X = data[['Citations per Paper', 'Papers per Faculty', 'Academic Reputation', 'Faculty Student Ratio']]
y = data['Success']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# API endpoints
@app.get("/predict")
def predict():
    try:
        data['Predicted Success'] = model.predict(X)
        return data[['University Name', 'Overall Score', 'Predicted Success']].to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/accuracy")
def get_accuracy():
    try:
        return {"accuracy": f"{accuracy:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving accuracy: {str(e)}")

# Handler for Vercel
handler = Mangum(app)