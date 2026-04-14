from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import json
# import requests

from pydantic import BaseModel, Field
import os
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your Vercel domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#####################################################################

class Listing(BaseModel):
    listing_type: str | None = None
    condition: str | None = None
    description: str | None = None
    price: float | None = None
    grade: str | None = None
    platform: str | None = None

class MarketplaceItem(BaseModel):
    item_name: str
    listings: List[Listing]

class Interpretation(BaseModel):
    plan: str = Field(description="Describe how you'll solve the problem.")
    reasoning_steps: List[str] = Field(description="Reasoning Steps")
    current_estimate: float
    current_high_range: float
    current_low_range: float
    current_trend: str = Field(description="'increasing' or 'decreasing' or 'steady'")
    evidence: List[str] = Field(description="List of Evidence Supporting Summary")
    assumptions: List[str] = Field(description="List of Assumptions for Summary")
    limitations: List[str] = Field(description="List of Limitations of Summary")
    alternative_interpretations: List[str] = Field(description="List of Alternative explanations")
        
print("AI Interpreter Loaded")

@app.get("/hello")
def hello():
    return {"message": "hello world!"}

@app.post("/interpret/")
async def interpret(item: MarketplaceItem):
    google_api_key = os.getenv('GOOGLE_API_KEY')
    client = genai.configure(api_key=google_api_key)

    prompt_template = """
        You are an expert in collectibles with a knowledge of market trends.
        Solve the following problem, reasoning step-by-step in a structured way. 

        Problem:
        Given the following data which details individual sales of the exact same collectible, generate a summary of the current market value of the collectible with the current price estimate, price range, and price trend. Include evidence, assumptions, limitations, and alternative interpretations in brief, listed format readable to average collectors.

        {{listing_data}}

        The values in the JSON correspond to the following:
        Condition: Grade 7, Grade 8, Grade 9
        Platform: eBay, Discogs, Etsy, Amazon.
        Date: MM-DD-YYYY.
        Actual Sold Price: $USD.

        Output your reasoning and solution in this exact JSON format:
        {{
        "plan": "Describe how you'll solve the problem.",
        "reasoning_steps": ["Step 1...", "Step 2...", "..."],
        "current_estimate": Number,
        "current_high_range": Number,
        "current_low_range": Number,
        "current_trend": String: "increasing" or "decreasing" or "steady",
        "evidence": ["Single Evidence Supporting Summary", "...", "Max 3"],
        "assumptions": ["Single Assumption for Summary", "...", "Max 3"],
        "limitations": ["Single Limitation of Summary", "...", "Max 3"],
        "alternative_interpretations": ["Single Alternative explanation", "...", "Max 3"]
        }}

        Important:
        - Keep the JSON strictly valid.
    
        """
    
    prompt = prompt_template.format(listing_data=json.dumps(item))

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        content=types.Part.from_text(text=prompt),
        config = {
            "response_mime_type": "application/json",
            "response_json_schema": Interpretation.model_json_schema()
        }
    )
    client.close()
    return {"message": response}


def interpret_test():
    item = {
        "item_name": "Ella Fitzgerald Record",
        "listings": [
            {
                "listing_type": "N/A",
                "condition": "VG",
                "description": "N/A",
                "price": 221.10,
                "platform": "ebay"
            },
            {
                "listing_type": "str",
                "condition": "VG",
                "description": "N/A",
                "price": 224.30,
                "platform": "discogs"
            },
            {
                "listing_type": "N/A",
                "condition": "G",
                "description": "N/A",
                "price": 200.10,
                "platform": "ebay"
            },
        ]
    }

    google_api_key = os.getenv('GOOGLE_API_KEY')
    client = genai.configure(api_key=google_api_key)

    prompt_template = """
        You are an expert in collectibles with a knowledge of market trends.
        Solve the following problem, reasoning step-by-step in a structured way. 

        Problem:
        Given the following data which details individual sales of the exact same collectible, generate a summary of the current market value of the collectible with the current price estimate, price range, and price trend. Include evidence, assumptions, limitations, and alternative interpretations in brief, listed format readable to average collectors.

        {{listing_data}}

        The values in the JSON correspond to the following:
        Condition: Grade 7, Grade 8, Grade 9
        Platform: eBay, Discogs, Etsy, Amazon.
        Date: MM-DD-YYYY.
        Actual Sold Price: $USD.

        Output your reasoning and solution in this exact JSON format:
        {{
        "plan": "Describe how you'll solve the problem.",
        "reasoning_steps": ["Step 1...", "Step 2...", "..."],
        "current_estimate": Number,
        "current_high_range": Number,
        "current_low_range": Number,
        "current_trend": String: "increasing" or "decreasing" or "steady",
        "evidence": ["Single Evidence Supporting Summary", "...", "Max 3"],
        "assumptions": ["Single Assumption for Summary", "...", "Max 3"],
        "limitations": ["Single Limitation of Summary", "...", "Max 3"],
        "alternative_interpretations": ["Single Alternative explanation", "...", "Max 3"]
        }}

        Important:
        - Keep the JSON strictly valid.
    """            
    
    prompt = prompt_template.format(listing_data=str(json.dumps(item)))
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=types.Part.from_text(text=prompt),
        config = {
            "response_mime_type": "application/json",
            "response_json_schema": Interpretation.model_json_schema()
        }
    )
    client.close()

    return response

    
# print(interpret_test())