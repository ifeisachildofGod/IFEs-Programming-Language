
from typing import Any
from dataclasses import dataclass

@dataclass
class ParsedCode:
    scope_data: dict[str, dict[str, "ParsedCode"]]
    type: str
    variables: dict[str, Any]
    data_type: "ParsedCode" = None
    parse_id: str = None
    
    def copy(self):
        return ParsedCode(scope_data=self.scope_data, type=self.type, variables=self.variables, data_type=self.data_type)
    
    def update(self, other: "ParsedCode"):
        self.scope_data = other.scope_data
        self.type = other.type
        self.variables = other.variables
        self.data_type = other.data_type
        self.parse_id = other.parse_id


