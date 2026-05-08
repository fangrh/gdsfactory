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
import re
import threading
from typing import Any

# GDS properties use integer keys. 1001 is reserved for SOURCE_PROP_KEY.
PROV_ID_PROP_KEY = 1002
PLACEMENT_PROP_KEY = 1004
SOURCE_TAG_INSTANCE_KEY = 1005  # Property key on Instance objects for source tags

# Module-level global counter ensures PROV_ID uniqueness across all trackers.
_global_id_lock = threading.Lock()
_global_next_id = 0


def _next_global_id() -> int:
    global _global_next_id
    with _global_id_lock:
        pid = _global_next_id
        _global_next_id += 1
        return pid


def _reset_global_id() -> None:
    """Reset the global ID counter and tracker dict. Called between builds."""
    global _global_next_id, _trackers_by_cell_index
    with _global_id_lock:
        _global_next_id = 0
    _trackers_by_cell_index = {}


# Module-level dict to store trackers by klayout cell index.
# kfactory's cell caching creates new Python wrappers for cached components,
# so _provenance_tracker on the Component object gets lost. Keying by
# cell_index (stable across wrapper replacements) survives this.
_trackers_by_cell_index: dict[int, ProvenanceTracker] = {}


def get_tracker(cell_index: int) -> ProvenanceTracker | None:
    return _trackers_by_cell_index.get(cell_index)


def set_tracker(cell_index: int, tracker: ProvenanceTracker) -> None:
    _trackers_by_cell_index[cell_index] = tracker


_GDSFACTORY_DIRS: tuple[str, ...] = (
    "gdsfactory",
    "kfactory",
    "klayout",
    "cachetools",
    "loguru",
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
                loop_index = _try_extract_loop_index(current, source_line)
                if loop_index is not None:
                    user_info["loop_index"] = loop_index
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


_FOR_LOOP_RE = re.compile(r"^for\s+([\w]+)")


def _try_extract_loop_index(frame, source_line: str) -> list[int] | None:
    """Best-effort extraction of loop iteration indices from a user frame.

    Scans backwards from the user frame's source line to find enclosing
    for-loop headers (``for VAR in range(...)``, ``for VAR in items``, etc.),
    then looks up each loop variable in the frame's locals.

    Uses indentation tracking to only accept for-loop headers that sit on
    the **enclosing block path** — i.e. lines whose indentation forms a
    strictly decreasing sequence when scanning backwards from the current
    line.  A for-loop whose body has already ended (e.g. same-indent code
    appears between it and the current line) is correctly skipped.

    Returns a list of integer indices (outermost first) or None if no
    enclosing integer-indexed loop is found.
    """
    explicit_index = frame.f_locals.get("_gds_provenance_loop_index")
    if isinstance(explicit_index, int):
        return [explicit_index]
    if (
        isinstance(explicit_index, list)
        and explicit_index
        and all(isinstance(value, int) for value in explicit_index)
    ):
        return explicit_index

    filepath = frame.f_code.co_filename
    lineno = frame.f_lineno

    indices: list[int] = []
    seen_vars: set[str] = set()

    # Build the list of "block boundary" lines — lines at a new minimum
    # indentation (strictly decreasing) as we scan backwards.  Only for-loop
    # headers that appear in this boundary list are truly enclosing.
    boundary_indents: list[int] = []  # decreasing indent levels
    current_min: int | None = None

    for offset in range(1, 30):
        raw_line = linecache.getline(filepath, lineno - offset)
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Stop at function / class / decorator boundary
        if stripped.startswith(("def ", "class ", "@")):
            break

        line_indent = len(raw_line) - len(raw_line.lstrip())

        # Only record lines that introduce a new (lower) indent level
        if current_min is None or line_indent < current_min:
            current_min = line_indent
            boundary_indents.append(line_indent)

            # If this boundary line is a for-loop, it encloses the current line
            m = _FOR_LOOP_RE.match(stripped)
            if m:
                var_name = m.group(1)
                if var_name not in seen_vars:
                    seen_vars.add(var_name)
                    val = frame.f_locals.get(var_name)
                    if isinstance(val, int):
                        indices.insert(0, val)

    return indices if indices else None


def tag_shapes_with_placement(kdb_cell, user_info: dict, instance_prov_id: int) -> None:
    """Tag all shapes in a cell (and descendants) with placement source info.

    When ``c << component`` is called, this propagates the placement call's
    source location (file, line, source_text) down to every shape inside the
    child component so that downstream consumers can attribute each shape
    to the user-level placement call rather than the internal polygon creation.
    """
    if user_info is None:
        return
    tag = json.dumps({
        "file": user_info["file"],
        "line": user_info["line"],
        "source_text": user_info["source_text"],
        "instance_prov_id": instance_prov_id,
    })
    _tag_cell_recursive(kdb_cell, tag, PLACEMENT_PROP_KEY)


def _tag_cell_recursive(cell, tag: str, prop_key: int = PLACEMENT_PROP_KEY) -> None:
    if cell.is_locked():
        cell.locked = False
    for li in range(cell.layout().layers()):
        if not cell.layout().is_valid_layer(li):
            continue
        for shape in cell.shapes(li).each():
            shape.set_property(prop_key, tag)
    for ci in cell.each_child_cell():
        child = cell.layout().cell(ci)
        if child is not None:
            _tag_cell_recursive(child, tag, prop_key)


class ProvenanceTracker:
    """Captures per-shape provenance for a Component build.

    Uses module-level global counter for unique PROV_IDs across all
    components in a build. write_sidecar() merges entries from child
    component trackers.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._child_trackers: list[ProvenanceTracker] = []

    def capture(self, component_name: str, element_type: str = "polygon") -> int:
        """Capture caller info and return a globally-unique provenance ID."""
        pid = _next_global_id()
        user_info = _find_user_frame()

        entry: dict[str, Any] = {
            "id": pid,
            "component": component_name,
            "element_type": element_type,
        }

        if user_info is not None:
            entry.update(user_info)
        else:
            entry.update({
                "file": "<unknown>",
                "line": 0,
                "function": "<unknown>",
                "source_text": "",
                "call_stack": [],
            })

        self._entries.append(entry)
        return pid

    def track_instance(
        self,
        parent_name: str,
        ref_name: str,
        instance_path: str,
        transform: str,
        columns: int = 1,
        rows: int = 1,
    ) -> int:
        """Track an add_ref placement with instance path (D13)."""
        pid = _next_global_id()
        user_info = _find_user_frame()

        entry: dict[str, Any] = {
            "id": pid,
            "component": parent_name,
            "element_type": "instance",
            "ref_component": ref_name,
            "instance_path": instance_path,
            "transform": transform,
        }

        if user_info is not None:
            entry.update({
                "file": user_info["file"],
                "line": user_info["line"],
                "function": user_info["function"],
                "source_text": user_info["source_text"],
                "call_stack": user_info["call_stack"],
            })
            if "loop_index" in user_info:
                entry["loop_index"] = user_info["loop_index"]
        else:
            entry.update({
                "file": "<unknown>",
                "line": 0,
                "function": "<unknown>",
                "source_text": "",
                "call_stack": [],
            })

        if columns > 1 or rows > 1:
            entry["loop_index"] = [columns, rows]

        self._entries.append(entry)
        return pid

    def add_child_tracker(self, child: ProvenanceTracker) -> None:
        """Register a child component's tracker for merging at write time."""
        if child is not self and child not in self._child_trackers:
            self._child_trackers.append(child)

    def _all_entries(self) -> list[dict[str, Any]]:
        """Collect entries from this tracker and all children, deduped by ID."""
        seen: set[int] = set()
        entries: list[dict[str, Any]] = []
        for entry in self._entries:
            if entry["id"] not in seen:
                seen.add(entry["id"])
                entries.append(entry)
        for child in self._child_trackers:
            for entry in child._all_entries():
                if entry["id"] not in seen:
                    seen.add(entry["id"])
                    entries.append(entry)
        entries.sort(key=lambda e: e["id"])
        return entries

    def get_sidecar(self) -> dict[str, Any]:
        """Return the provenance data as a dict for JSON serialization."""
        return {"version": 1, "entries": self._all_entries()}

    def write_sidecar(self, gds_path: str | pathlib.Path) -> pathlib.Path:
        """Write .provenance.json sidecar next to the GDS file."""
        gds_path = pathlib.Path(gds_path)
        sidecar_path = gds_path.with_suffix(".provenance.json")
        sidecar_path.write_text(
            json.dumps(self.get_sidecar(), indent=2), encoding="utf-8"
        )
        return sidecar_path
