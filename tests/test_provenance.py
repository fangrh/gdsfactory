"""Tests for provenance tracking (gdsfactory/provenance.py)."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import gc

import pytest


@pytest.fixture(autouse=True)
def _clean_env():
    """Ensure GDS_PROVENANCE is unset unless a test sets it explicitly."""
    old = os.environ.pop("GDS_PROVENANCE", None)
    yield
    if old is not None:
        os.environ["GDS_PROVENANCE"] = old
    else:
        os.environ.pop("GDS_PROVENANCE", None)


def test_no_provenance_by_default():
    """Without GDS_PROVENANCE=1, no sidecar is generated."""
    import gdsfactory as gf

    c = gf.components.rectangle(size=(10, 5), layer="WG")
    with tempfile.TemporaryDirectory() as tmpdir:
        gdspath = c.write_gds(gdsdir=tmpdir)
        sidecar = gdspath.with_suffix(".provenance.json")
        assert not sidecar.exists(), "Sidecar should not exist without GDS_PROVENANCE=1"


def test_sidecar_generated_with_env():
    """With GDS_PROVENANCE=1, sidecar is generated with valid entries."""
    import gdsfactory as gf

    os.environ["GDS_PROVENANCE"] = "1"

    c = gf.Component("prov_test")
    c.add_polygon([(0, 0), (10, 0), (10, 5), (0, 5)], layer="WG")

    with tempfile.TemporaryDirectory() as tmpdir:
        gdspath = c.write_gds(gdsdir=tmpdir)
        sidecar = gdspath.with_suffix(".provenance.json")
        assert sidecar.exists(), "Sidecar should be generated with GDS_PROVENANCE=1"

        data = json.loads(sidecar.read_text())
        assert data["version"] == 1
        assert len(data["entries"]) > 0, "Should have at least one provenance entry"

        entry = data["entries"][0]
        assert "id" in entry
        assert "file" in entry
        assert "line" in entry
        assert "function" in entry
        assert "element_type" in entry
        assert entry["element_type"] == "polygon"


def test_prov_id_on_shapes():
    """Shapes in the GDS have PROV_ID property matching sidecar entries."""
    import gdsfactory as gf
    import klayout.db as kdb

    os.environ["GDS_PROVENANCE"] = "1"

    c = gf.Component("prov_shapes_test")
    c.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], layer="WG")
    c.add_polygon([(10, 0), (15, 0), (15, 5), (10, 5)], layer="WG")

    with tempfile.TemporaryDirectory() as tmpdir:
        gdspath = c.write_gds(gdsdir=tmpdir)
        sidecar = gdspath.with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text())
        prov_ids_in_sidecar = {e["id"] for e in data["entries"]}

        # Read GDS and check PROV_ID properties on shapes
        layout = kdb.Layout()
        layout.read(str(gdspath))
        cell = layout.cell(0)

        prov_ids_on_shapes: set[int] = set()
        for li in layout.layer_indexes():
            for shape in cell.shapes(li).each():
                prop = shape.property(1002)
                if prop is not None:
                    prov_ids_on_shapes.add(int(prop))

        assert prov_ids_on_shapes, "Should find PROV_ID properties on shapes"
        assert prov_ids_on_shapes.issubset(prov_ids_in_sidecar), (
            f"Shape PROV_IDs {prov_ids_on_shapes} should be in sidecar {prov_ids_in_sidecar}"
        )


def test_add_ref_tracking():
    """add_ref creates distinct provenance entries for each placement (D13)."""
    import gdsfactory as gf

    os.environ["GDS_PROVENANCE"] = "1"

    child = gf.components.rectangle(size=(4, 2), layer="WG")
    parent = gf.Component("ref_tracking_test")
    ref1 = parent << child
    ref1.move((0, 0))
    ref2 = parent << child
    ref2.move((20, 0))

    with tempfile.TemporaryDirectory() as tmpdir:
        gdspath = parent.write_gds(gdsdir=tmpdir)
        sidecar = gdspath.with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text())

        instances = [e for e in data["entries"] if e["element_type"] == "instance"]
        assert len(instances) == 2, (
            f"Should have 2 instance entries for 2 add_ref calls, got {len(instances)}"
        )

        paths = {e["instance_path"] for e in instances}
        assert len(paths) == 2, "Each instance should have a distinct instance path"


def test_frame_cleanup():
    """No lingering frame references after provenance capture (D10)."""
    os.environ["GDS_PROVENANCE"] = "1"

    from gdsfactory.provenance import _find_user_frame

    gc.collect()
    gc.disable()
    try:
        result = _find_user_frame()
        assert result is not None, "Should find user frame from test"
        gc.collect()
        # If frames were properly cleaned, gc.garbage shouldn't grow
        assert len(gc.garbage) < 10, (
            f"Too many garbage objects ({len(gc.garbage)}), possible frame leak"
        )
    finally:
        gc.enable()


def test_nested_components():
    """Sub-component shapes get provenance in the top-level sidecar."""
    import gdsfactory as gf

    os.environ["GDS_PROVENANCE"] = "1"

    child = gf.Component("prov_child")
    child.add_polygon([(0, 0), (4, 0), (4, 2), (0, 2)], layer="WG")

    parent = gf.Component("nested_test")
    parent << child
    parent.add_polygon([(0, 0), (8, 0), (8, 4), (0, 4)], layer="WG")

    with tempfile.TemporaryDirectory() as tmpdir:
        gdspath = parent.write_gds(gdsdir=tmpdir)
        sidecar = gdspath.with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text())

        assert len(data["entries"]) > 0, "Should have entries from nested component"
        # Should have at least: child polygon, parent polygon, parent instance
        element_types = {e["element_type"] for e in data["entries"]}
        assert "polygon" in element_types, "Should have polygon entries"
        assert "instance" in element_types, "Should have instance entry for add_ref"


def test_label_provenance():
    """Labels get provenance entries."""
    import gdsfactory as gf

    os.environ["GDS_PROVENANCE"] = "1"

    c = gf.Component("label_prov_test")
    c.add_polygon([(0, 0), (10, 0), (10, 5), (0, 5)], layer="WG")
    c.add_label(text="test_label", position=(5, 2.5), layer="TEXT")

    with tempfile.TemporaryDirectory() as tmpdir:
        gdspath = c.write_gds(gdsdir=tmpdir)
        sidecar = gdspath.with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text())

        labels = [e for e in data["entries"] if e["element_type"] == "label"]
        assert len(labels) == 1, f"Should have 1 label entry, got {len(labels)}"
        assert labels[0]["source_text"] != "", "Label entry should have source text"
