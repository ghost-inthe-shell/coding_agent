"""JSON-compatible types shared by persisted protocols."""

from typing import Dict, List, TypeAlias, Union

JsonPrimitive: TypeAlias = Union[None, bool, int, float, str]
JsonValue: TypeAlias = Union[JsonPrimitive, List["JsonValue"], Dict[str, "JsonValue"]]
JsonObject: TypeAlias = Dict[str, JsonValue]

