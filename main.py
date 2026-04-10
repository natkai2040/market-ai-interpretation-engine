from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# import json
# import requests
from pydantic import BaseModel
# import os
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

print("AI Interpreter Loaded")

@app.post("/interpret/")
async def interpret(item: MarketplaceItem):
    return {"message": "received item " + item.item_name}

