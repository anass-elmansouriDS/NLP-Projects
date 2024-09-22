from fastapi import FastAPI
from keyblend import KeyBlender
import pickle
# Initialize the FastAPI app
app = FastAPI()

# Define a route (path) and a function that handles requests to that path
@app.get("/")
def read_root():
    return {"message": "Welcome to KeyBlender Recommender for content-product recommendation!"}

# Another route that accepts a parameter
@app.get("/predict")
def predict(content : str) :
    recommender=KeyBlender()
    recommendations=recommender.recommend(content)
    return list(recommendations.keys())[:5]