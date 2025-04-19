from pydantic import BaseModel, Field, field_validator
from typing import List

class LandmarkRequest(BaseModel):
    landmarks: List[List[List[float]]] = Field(
        ...,
        description='List of frames, each containing a list of landmarks with x, y, z coordinates. Shape: [frames, 180, 3].'
    )
    top_n: int = Field(5, ge=1, le=10, description='Number of top glosses to return')

    @field_validator('landmarks')
    def validate_landmarks(cls, v):
        if not isinstance(v, list) or not v: raise ValueError('Landmarks must be a non-empty list')
        for frame in v:
            if len(frame) != 180 or not all(len(lm) == 3 for lm in frame):
                raise ValueError('Each frame must have 180 landmarks with 3 coordinates')
        return v

class GlossPrediction(BaseModel):
    gloss: str
    score: float = Field(..., ge=0.0, le=1.0, description='Softmax score for the gloss')

class PredictionResponse(BaseModel):
    predictions: List[GlossPrediction] = Field(..., description='Top N gloss predictions with scores')
    