"""Compile a JSON Schema into a GBNF grammar.

llama.cpp can constrain decoding to a grammar, which is the strongest guarantee
of well-formed output available for a model running entirely on the local
machine — stronger than prompting, and it works on models that have never seen a
tool-call in their life. The catch is that llama.cpp wants GBNF, not JSON Schema.

This module bridges that gap for the subset of JSON Schema that pydantic actually
emits: typed objects with required and optional properties, arrays, the scalar
types, enums, ``const``, ``anyOf``/``oneOf`` unions, and ``$ref`` into ``$defs``.
The same pydantic model therefore drives every structured-output path, schema or
grammar, with no second source of truth.

Limitations are deliberate and documented: property order is fixed, and an object
with no required properties is treated as all-required. Both keep the grammar
finite and are harmless for the report schemas this project generates.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

_PRIMITIVE_BODIES: dict[str, str] = {
    "ws": r"[ \t\n]*",
    "hex": r"[0-9a-fA-F]",
    "char": r'[^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)',
    "string": r'"\"" char* "\""',
    "integer": r'"-"? ("0" | [1-9] [0-9]*)',
    "number": r'"-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] ("-" | "+")? [0-9]+)?',
    "boolean": r'"true" | "false"',
    "null": r'"null"',
    "value": "object | array | string | number | boolean | null",
    "object": r'"{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"',
    "array": r'"[" ws (value (ws "," ws value)*)? ws "]"',
}

_PRIMITIVE_DEPS: dict[str, tuple[str, ...]] = {
    "ws": (),
    "hex": (),
    "char": ("hex",),
    "string": ("char",),
    "integer": (),
    "number": (),
    "boolean": (),
    "null": (),
    "value": ("object", "array", "string", "number", "boolean", "null"),
    "object": ("ws", "string", "value"),
    "array": ("ws", "value"),
}


def model_to_gbnf(model_type: type[BaseModel]) -> str:
    """Compile a pydantic model's JSON Schema into a GBNF grammar."""
    return schema_to_gbnf(model_type.model_json_schema())


def schema_to_gbnf(schema: dict[str, Any]) -> str:
    """Compile a JSON Schema dict into a GBNF grammar string."""
    return _Compiler(schema).compile()


class _Compiler:
    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema
        self._defs: dict[str, Any] = schema.get("$defs", {})
        self._rules: dict[str, str] = {}
        self._ref_cache: dict[str, str] = {}
        self._used_primitives: set[str] = {"ws"}
        self._counter = 0

    def compile(self) -> str:
        root_expr = self._node(self._schema)
        lines = [f"root ::= ws {root_expr} ws"]
        lines.extend(f"{name} ::= {body}" for name, body in self._rules.items())
        for name in self._primitive_closure():
            lines.append(f"{name} ::= {_PRIMITIVE_BODIES[name]}")
        return "\n".join(lines) + "\n"

    # -- node dispatch ---------------------------------------------------

    def _node(self, node: dict[str, Any]) -> str:
        if "$ref" in node:
            return self._ref(node["$ref"])
        if "const" in node:
            return _gbnf_literal_json(node["const"])
        if "enum" in node:
            return "(" + " | ".join(_gbnf_literal_json(v) for v in node["enum"]) + ")"
        union = node.get("anyOf") or node.get("oneOf")
        if union:
            return "(" + " | ".join(self._node(option) for option in union) + ")"

        node_type = _scalar_type(node)
        if node_type == "object" or "properties" in node:
            return self._object(node)
        if node_type == "array":
            return self._array(node)
        if node_type in ("string", "integer", "number", "boolean", "null"):
            return self._primitive(node_type)
        return self._primitive("value")

    # -- composite types -------------------------------------------------

    def _object(self, node: dict[str, Any]) -> str:
        properties: dict[str, Any] = node.get("properties", {})
        if not properties:
            return self._primitive("object")

        required = node.get("required", [])
        required_keys = [key for key in properties if key in required]
        optional_keys = [key for key in properties if key not in required]
        if not required_keys:  # keep the grammar finite: demand everything
            required_keys, optional_keys = list(properties), []

        self._use("ws")
        parts = ['"{"', "ws", self._pair(required_keys[0], properties[required_keys[0]])]
        for key in required_keys[1:]:
            parts += ["ws", '","', "ws", self._pair(key, properties[key])]
        for key in optional_keys:
            parts.append(f'(ws "," ws {self._pair(key, properties[key])})?')
        parts += ["ws", '"}"']

        name = self._fresh("obj")
        self._rules[name] = " ".join(parts)
        return name

    def _array(self, node: dict[str, Any]) -> str:
        items = node.get("items")
        inner = self._node(items) if isinstance(items, dict) else self._primitive("value")
        self._use("ws")
        name = self._fresh("arr")
        self._rules[name] = f'"[" ws ({inner} (ws "," ws {inner})*)? ws "]"'
        return name

    def _pair(self, key: str, schema: dict[str, Any]) -> str:
        return f'{_gbnf_key(key)} ws ":" ws {self._node(schema)}'

    def _ref(self, ref: str) -> str:
        if ref in self._ref_cache:
            return self._ref_cache[ref]
        name = ref.split("/")[-1]
        target = self._defs.get(name)
        if target is None:
            return self._primitive("value")
        expr = self._node(target)
        self._ref_cache[ref] = expr
        return expr

    # -- primitives & bookkeeping ----------------------------------------

    def _primitive(self, name: str) -> str:
        self._use(name)
        return name

    def _use(self, name: str) -> None:
        self._used_primitives.add(name)

    def _fresh(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def _primitive_closure(self) -> list[str]:
        pending = list(self._used_primitives)
        closure: set[str] = set()
        while pending:
            name = pending.pop()
            if name in closure:
                continue
            closure.add(name)
            pending.extend(_PRIMITIVE_DEPS.get(name, ()))
        return [name for name in _PRIMITIVE_BODIES if name in closure]


def _scalar_type(node: dict[str, Any]) -> str | None:
    node_type = node.get("type")
    if isinstance(node_type, list):  # e.g. ["string", "null"]
        for candidate in node_type:
            if candidate != "null":
                return str(candidate)
        return "null"
    return node_type if isinstance(node_type, str) else None


def _gbnf_key(key: str) -> str:
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'"\\"{escaped}\\""'


def _gbnf_literal_json(value: Any) -> str:
    rendered = json.dumps(value)
    escaped = rendered.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
