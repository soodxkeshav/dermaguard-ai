"""
Data Schemas for DermaGuard AI FastAPI Backend
"""

from typing import Optional, List, Dict, Any

class PredictionRequest:
    def __init__(self, image_base64: str, skin_tone: Optional[str] = "Type_III", notes: Optional[str] = None):
        self.image_base64 = image_base64
        self.skin_tone = skin_tone
        self.notes = notes

class PredictionResponse:
    def __init__(
        self,
        prediction_id: str,
        predicted_category: str,
        risk_level: str,
        confidence: float,
        explanation: str,
        recommendation: str,
        timestamp: str
    ):
        self.prediction_id = prediction_id
        self.predicted_category = predicted_category
        self.risk_level = risk_level
        self.confidence = confidence
        self.explanation = explanation
        self.recommendation = recommendation
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "predicted_category": self.predicted_category,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp
        }
