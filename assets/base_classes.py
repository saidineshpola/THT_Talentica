# ------------- Pydantic Schemas (Data Models) ---------------
import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Item(BaseModel):
    name: str
    qty: Optional[int] = None
    price: Optional[float] = None

class QAPair(BaseModel):
    question: str
    answer: str

class BillData(BaseModel):
    id: str
    original_filename: Optional[str] = None
    store_name: Optional[str] = None
    date: Optional[str] = None
    items: List[Item] = Field(default_factory=list)
    total: Optional[float] = None
    gst: Optional[float] = None
    raw_text: Optional[str] = None
    q_and_a: List[QAPair] = Field(default_factory=list)

class ChatResponse(BaseModel):
    reply: str

class QueryResponse(BaseModel):
    answer: str
    bill_id: Optional[str] = None


# Application State
class AppState:
    def __init__(self):
        self.bills_store: Dict[str, BillData] = {} # Stores processed BillData objects, keyed by bill.id
        self._next_bill_id_counter: int = 0
        self._lock = asyncio.Lock()

    async def get_next_id(self) -> str:
        async with self._lock:
            self._next_bill_id_counter += 1
            return f"bill_{self._next_bill_id_counter}"

    async def find_bill(self, identifier: str) -> Optional[BillData]:
        """Finds a bill by its internal ID or original_filename."""
        if identifier in self.bills_store:
            return self.bills_store[identifier]
        for bill in self.bills_store.values():
            if bill.original_filename == identifier:
                return bill
        return None
