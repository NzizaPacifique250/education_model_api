from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Education in Africa - School Life Expectancy Predictor",
    description="API to predict school life expectancy based on government education expenditure",
    version="1.0.0"
)

# CRITICAL: Add CORS middleware BEFORE any routes
# This must be one of the first things after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - necessary for Flutter
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Load model and scaler at startup
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✓ Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# Pydantic model for input validation
class PredictionInput(BaseModel):
    """
    Input schema for school life expectancy prediction.
    
    All expenditure values should be percentages of GDP.
    """
    year: int = Field(
        ..., 
        ge=2000, 
        le=2030,
        description="Year of data (between 2000 and 2030)"
    )
    exp_primary_gdp: float = Field(
        ..., 
        ge=0.0, 
        le=10.0,
        description="Government expenditure on primary education as % of GDP (0-10%)"
    )
    exp_secondary_gdp: float = Field(
        ..., 
        ge=0.0, 
        le=10.0,
        description="Government expenditure on secondary education as % of GDP (0-10%)"
    )
    exp_tertiary_gdp: float = Field(
        ..., 
        ge=0.0, 
        le=10.0,
        description="Government expenditure on tertiary education as % of GDP (0-10%)"
    )
    total_exp_gdp: float = Field(
        ..., 
        ge=0.0, 
        le=25.0,
        description="Total government expenditure on education as % of GDP (0-25%)"
    )
    
    @validator('total_exp_gdp')
    def validate_total_expenditure(cls, v, values):
        """Ensure total expenditure is reasonable compared to components"""
        if 'exp_primary_gdp' in values and 'exp_secondary_gdp' in values and 'exp_tertiary_gdp' in values:
            component_sum = values['exp_primary_gdp'] + values['exp_secondary_gdp'] + values['exp_tertiary_gdp']
            if v < component_sum * 0.8 or v > component_sum * 1.5:
                raise ValueError(
                    f"Total expenditure ({v}%) should be close to sum of components ({component_sum:.2f}%)"
                )
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "year": 2020,
                "exp_primary_gdp": 2.5,
                "exp_secondary_gdp": 1.8,
                "exp_tertiary_gdp": 0.7,
                "total_exp_gdp": 5.0
            }
        }

# Pydantic model for output
class PredictionOutput(BaseModel):
    """Output schema for prediction result"""
    predicted_school_life_expectancy: float = Field(
        ..., 
        description="Predicted school life expectancy in years"
    )
    input_data: PredictionInput
    model_info: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_school_life_expectancy": 12.45,
                "input_data": {
                    "year": 2020,
                    "exp_primary_gdp": 2.5,
                    "exp_secondary_gdp": 1.8,
                    "exp_tertiary_gdp": 0.7,
                    "total_exp_gdp": 5.0
                },
                "model_info": {
                    "model_type": "Random Forest",
                    "features": ["year", "exp_primary_gdp", "exp_secondary_gdp", 
                                "exp_tertiary_gdp", "total_exp_gdp"]
                }
            }
        }

# Root endpoint
@app.get("/")
def read_root():
    """Welcome endpoint with API information"""
    return {
        "message": "Education in Africa - School Life Expectancy Prediction API",
        "mission": "Predict school life expectancy based on government education investment",
        "endpoints": {
            "GET /": "This welcome message",
            "POST /predict": "Make a prediction",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation"
        },
        "version": "1.0.0",
        "cors_enabled": True
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if the API and model are loaded correctly"""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "cors_enabled": True
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Predict school life expectancy based on government education expenditure.
    
    The model predicts the expected number of years a child will spend in school
    from primary to tertiary education, based on government investment patterns.
    """
    # Check if model is loaded
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500, 
            detail="Model or scaler not loaded. Please check server configuration."
        )
    
    try:
        # Prepare input array in correct feature order
        input_array = np.array([[
            input_data.year,
            input_data.exp_primary_gdp,
            input_data.exp_secondary_gdp,
            input_data.exp_tertiary_gdp,
            input_data.total_exp_gdp
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Round to 2 decimal places
        prediction = round(float(prediction), 2)
        
        # Prepare response
        return PredictionOutput(
            predicted_school_life_expectancy=prediction,
            input_data=input_data,
            model_info={
                "model_type": type(model).__name__,
                "features": [
                    "year",
                    "exp_primary_gdp",
                    "exp_secondary_gdp",
                    "exp_tertiary_gdp",
                    "total_exp_gdp"
                ],
                "unit": "years"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# OPTIONS endpoint for CORS preflight (explicit handler)
@app.options("/predict")
async def predict_options():
    """Handle OPTIONS request for CORS preflight"""
    return {"message": "OK"}

# Example endpoint to get valid input ranges
@app.get("/input-ranges")
def get_input_ranges():
    """Get valid ranges for input parameters"""
    return {
        "year": {
            "min": 2000,
            "max": 2030,
            "type": "integer",
            "description": "Year of data"
        },
        "exp_primary_gdp": {
            "min": 0.0,
            "max": 10.0,
            "type": "float",
            "unit": "% of GDP",
            "description": "Government expenditure on primary education"
        },
        "exp_secondary_gdp": {
            "min": 0.0,
            "max": 10.0,
            "type": "float",
            "unit": "% of GDP",
            "description": "Government expenditure on secondary education"
        },
        "exp_tertiary_gdp": {
            "min": 0.0,
            "max": 10.0,
            "type": "float",
            "unit": "% of GDP",
            "description": "Government expenditure on tertiary education"
        },
        "total_exp_gdp": {
            "min": 0.0,
            "max": 25.0,
            "type": "float",
            "unit": "% of GDP",
            "description": "Total government expenditure on education",
            "note": "Should be approximately equal to sum of primary + secondary + tertiary"
        }
    }

# Run with: uvicorn prediction:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)