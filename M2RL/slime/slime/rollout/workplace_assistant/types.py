from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

RESPOND_ACTION_NAME = "respond"

class Action(BaseModel):
    name: str
    type: str
    arguments: Dict[str, Any]

