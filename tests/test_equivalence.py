"""Equivalence with the Robust Weight Transfer Blender add-on on recorded production data (not in this repository).

Runs only when WEIGHTXFER_ADDON_IO points to a recorded ADDON_IO.npz (format: tools/equivalence.py); with
WEIGHTXFER_SAVED (the pipeline's weights.npz) it also checks the weights after the pipeline's top-4 step."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

IO = os.environ.get("WEIGHTXFER_ADDON_IO")


@pytest.mark.skipif(not IO, reason="set WEIGHTXFER_ADDON_IO (recorded add-on inputs and outputs) to run")
def test_matches_the_addon():
    import equivalence
    args = [IO] + (["--saved", os.environ["WEIGHTXFER_SAVED"]] if os.environ.get("WEIGHTXFER_SAVED") else [])
    res = equivalence.main(args)
    for p, r in res["parts"].items():
        assert r["fallback"] is None, p
        assert r["raw"]["max"] <= 1e-6, (p, r["raw"])      # float32 storage of the add-on's weights
        if "saved" in r:
            assert r["saved"]["max"] <= 1e-6, (p, r["saved"])
