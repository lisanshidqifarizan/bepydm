import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

data = pd.read_csv('topuniversities.csv')
data = data[['University Name', 'Overall Score', 'Citations per Paper', 'Papers per Faculty', 'Academic Reputation', 
             'Faculty Student Ratio', 'Staff with PhD', 'International Research Center', 
             'International Students', 'Outbound Exchange', 'Inbound Exchange', 
             'International Faculty', 'Employer Reputation']]
data = data.dropna()
data['Success'] = (data['Overall Score'] >= 50).astype(int)

X = data.drop(columns=['Overall Score', 'Success', 'University Name'])
y = data['Success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
classification_report_str = classification_report(y_test, y_pred)

data['Predicted Success'] = model.predict(X)

data.to_csv('processed_topuniversities.csv', index=False)

@app.get("/predict")
def get_predictions():
    try:
        processed_data = pd.read_csv('processed_topuniversities.csv')
        predictions = processed_data[['University Name', 'Overall Score', 'Predicted Success']].to_dict(orient='records')
        return {"predictions": predictions}
    except FileNotFoundError:
        return {"error": "Processed data file not found. Please train the model first."}

@app.get("/accuracy")
def get_accuracy():
    return {"accuracy": f"{accuracy:.2f}%"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)