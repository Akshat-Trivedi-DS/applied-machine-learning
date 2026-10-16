"""Pydantic request/response schemas for the credit risk API.

Field-level validation (ge/le bounds) rejects obviously malformed
input before it ever reaches the model, e.g. a negative age or an
EDUCATION code outside the documented 1-4 range.
"""

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    limit_bal: float = Field(..., gt=0, description="Credit limit (NT dollar)")
    sex: int = Field(..., ge=1, le=2, description="1 = male, 2 = female")
    education: int = Field(..., ge=1, le=4, description="1=grad school, 2=university, 3=high school, 4=other")
    marriage: int = Field(..., ge=1, le=3, description="1=married, 2=single, 3=other")
    age: int = Field(..., ge=18, le=100)

    pay_1: int = Field(..., ge=-2, le=8, description="Repayment status, most recent month")
    pay_2: int = Field(..., ge=-2, le=8)
    pay_3: int = Field(..., ge=-2, le=8)
    pay_4: int = Field(..., ge=-2, le=8)
    pay_5: int = Field(..., ge=-2, le=8)
    pay_6: int = Field(..., ge=-2, le=8)

    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    pay_amt1: float = Field(..., ge=0)
    pay_amt2: float = Field(..., ge=0)
    pay_amt3: float = Field(..., ge=0)
    pay_amt4: float = Field(..., ge=0)
    pay_amt5: float = Field(..., ge=0)
    pay_amt6: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "limit_bal": 200000, "sex": 2, "education": 2, "marriage": 1, "age": 35,
                "pay_1": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
                "bill_amt1": 50000, "bill_amt2": 48000, "bill_amt3": 46000,
                "bill_amt4": 44000, "bill_amt5": 42000, "bill_amt6": 40000,
                "pay_amt1": 3000, "pay_amt2": 3000, "pay_amt3": 3000,
                "pay_amt4": 3000, "pay_amt5": 3000, "pay_amt6": 3000,
            }
        }
    }


class PredictionResponse(BaseModel):
    default_probability: float
    risk_category: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
