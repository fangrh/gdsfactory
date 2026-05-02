"""Provenance tracking for per-shape source attribution.

When GDS_PROVENANCE=1 is set, captures the user-code call site for each
shape added to a Component and writes a .provenance.json sidecar alongside
the GDS file. Each shape gets a PROV_ID GDS property mapping to an entry
in the sidecar.
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import linecache
from typing import Any

# GDS properties use integer keys. 1001 is reserved for SOURCE_PROP_KEY.
PROV_ID_PROP_KEY = 1002


_GDSFACTORY_DIRS: tuple[str, ...] = (
    "gdsfactory",
    "kfactory",
    "klayout",
)


def _is_internal_frame(filepath: str) -> bool:
    """Return True if the frame is inside gdsfactory/kfactory/klayout internals."""
    parts = pathlib.Path(filepath).parts
    for d in _GDSFACTORY_DIRS:
        if d in parts:
            return True
    return False


def _find_user_frame() -> dict[str, Any] | None:
    """Walk the call stack past gdsfactory internals to find user code.

    Returns a dict with file, line, function, source_text, call_stack, or None.
    Cleans up frame references to avoid cycles (D10).
    """
    frame = inspect.currentframe()
    if frame is None:
        return None

    user_info: dict[str, Any] | None = None
    try:
        current = frame.f_back  # skip _find_user_frame itself
        frames_to_clean: list[Any] = [frame, current]

        while current is not None:
            filepath = current.f_code.co_filename
            if not _is_internal_frame(filepath):
                # Found user code
                source_line = linecache.getline(filepath, current.f_lineno).strip()
                call_stack: list[str] = []
                walker = current.f_back
                while walker is not None:
                    wp = walker.f_code.co_filename
                    if not _is_internal_frame(wp):
                        call_stack.append(
                            f"{pathlib.Path(wp).name}:{walker.f_lineno}"
                            f" in {walker.f_code.co_name}"
                        )
                    walker = walker.f_back
                    frames_to_clean.append(walker)

                user_info = {
                    "file": filepath,
                    "line": current.f_lineno,
                    "function": current.f_code.co_name,
                    "source_text": source_line,
                    "call_stack": call_stack,
                }
                break
            current = current.f_back
            frames_to_clean.append(current)
    finally:
        # D10: clean up frame references to break reference cycles
        for f in frames_to_clean:
            if f is not None:
                del f
        del frame
        del frames_to_clean

    return user_info


class ProvenanceTracker:
    """Captures per-shape provenance for a Component build.

    One tracker per write_gds call, stored on the top-level Component.
    Child components get the same tracker via propagation in add_ref.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._next_id: int = 0

    def capture(self, component_name: str, element_type: str = "polygon") -> int:
        """Capture caller info and return a provenance ID.

        Walks the call stack to find user code, records file/line/function.
        """
        user_info = _find_user_frame()
        if user_info is None:
            pid = self._next_id
            self._next_id += 1
            self._entries.append({
                "id": pid,
                "component": component_name,
                "element_type": element_type,
                "file": "<unknown>",
                "line": 0,
                "function": "<unknown>",
                "source_text": "",
                "call_stack": [],
            })
            return pid

        pid = self._next_id
        self._next_id += 1
        user_info["id"] = pid
        user_info["component"] = component_name
        user_info["element_type"] = element_type
        self._entries.append(user_info)
        return pid

    def track_instance(
        self,
        parent_name: str,
        ref_name: str,
        instance_path: str,
        transform: str,
    ) -> int:
        """Track an add_ref placement with instance path (D13).

        Each reference placement gets its own provenance entry, even
        if the same cell is referenced multiple times.
        """
        user_info = _find_user_frame()
        pid = self._next_id
        self._next_id += 1

        entry: dict[str, Any] = {
            "id": pid,
            "component": parent_name,
            "element_type": "instance",
            "ref_component": ref_name,
            "instance_path": instance_path,
            "transform": transform,
        }

        if user_info is not None:
            entry["file"] = user_info["file"]
            entry["line"] = user_info["line"]
            entry["function"] = user_info["function"]
            entry["source_text"] = user_info["source_text"]
            entry["call_stack"] = user_info["call_stack"]
        else:
            entry["file"] = "<unknown>"
            entry["line"] = 0
            entry["function"] = "<unknown>"
            entry["source_text"] = ""
            entry["call_stack"] = []

        self._entries.append(entry)
        return pid

    def get_sidecar(self) -> dict[str, Any]:
        """Return the provenance data as a dict for JSON serialization."""
        return {"version": 1, "entries": self._entries}

    def write_sidecar(self, gds_path: str | pathlib.Path) -> pathlib.Path:
        """Write .provenance.json sidecar next to the GDS file."""
        gds_path = pathlib.Path(gds_path)
        sidecar_path = gds_path.with_suffix(".provenance.json")
        sidecar_path.write_text(
            json.dumps(self.get_sidecar(), indent=2), encoding="utf-8"
        )
        return sidecar_path
