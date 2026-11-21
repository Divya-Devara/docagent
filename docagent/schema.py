"""Schema: the contract for one extracted receipt."""
from pydantic import BaseModel, Field


class Receipt(BaseModel):
    company: str = Field(min_length=1, description="Seller's business name as printed, e.g. ending in SDN BHD")
    date: str = Field(min_length=1, description="Purchase date only, no time, as printed, e.g. 25/12/2018")
    address: str = Field(min_length=1, description="Seller's full address, all lines joined with single spaces")
    total: float = Field(ge=0, description="Final amount paid after rounding, a number without currency")
    