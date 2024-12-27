from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mangum import Mangum
import logging
import os

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if CSV file exists
if not os.path.exists('topuniversities.csv'):
    raise FileNotFoundError("The file 'topuniversities.csv' is missing. Please ensure it exists in the project directory.")

# Sample dataset and model
data = pd.read_csv('topuniversities.csv')
data = data[['University Name', 'Overall Score', 'Citations per Paper', 'Papers per Faculty', 'Academic Reputation',
             'Faculty Student Ratio', 'Staff with PhD', 'International Research Center',
             'International Students', 'Outbound Exchange', 'Inbound Exchange',
             'International Faculty', 'Employer Reputation']]

# Process data
data = data.dropna()
data['Success'] = (data['Overall Score'] >= 50).astype(int)
X = data.drop(columns=['Overall Score', 'Success', 'University Name'])
y = data['Success']

def train_model():
    global model, accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    logger.info(f"Model trained with accuracy: {accuracy:.2f}%")

    # Save processed data
    data['Predicted Success'] = model.predict(X)
    data.to_csv('processed_topuniversities.csv', index=False)

# Train the model initially
train_model()

# API endpoints
@app.get("/predict")
def predict():
    try:
        processed_data = pd.read_csv('processed_topuniversities.csv')
        return processed_data[['University Name', 'Overall Score', 'Predicted Success']].to_dict(orient='records')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Processed data file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/accuracy")
def get_accuracy():
    try:
        return {"accuracy": f"{accuracy:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving accuracy: {str(e)}")

@app.post("/retrain")
def retrain():
    try:
        train_model()
        return {"message": "Model retrained successfully", "accuracy": f"{accuracy:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during retraining: {str(e)}")

# Handler for Vercel
handler = Mangum(app)