from pydantic import BaseModel


class ParkingSlotResponse(BaseModel):
    probability: float
    slot: str
