"""GroundingDINO autolabel: pure ROI math + orchestration with a stub labeler.

No torch/transformers here — the heavy model is injected, so these stay unit-fast. The
real model run is the manual/heavy step (needs the 'autolabel' extra + a model download).
"""

import io

import pytest
from PIL import Image

from tensor_factory.geometry import BBox
from tensor_factory_studio.autolabel import autolabel, map_box_to_full, roi_from_box

# Detection is a plain dataclass (no torch); only GroundingDinoAutoLabeler.__init__ is heavy.
from tensor_factory_synth.autolabel import Detection  # noqa: E402


def _png(size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (123, 222, 64)).save(buf, "PNG")
    return buf.getvalue()


class StubLabeler:
    def __init__(self, dets):
        self.dets = dets
        self.seen = []  # (image size, features) per call

    def label(self, image, features):
        self.seen.append((image.size, list(features)))
        return self.dets


@pytest.mark.unit
def test_roi_from_box_expands_by_margin_and_clamps():
    # box [0.4,0.4,0.6,0.6], margin 0.5 of the 0.2 side = 0.1 each way -> [0.3,0.3,0.7,0.7]
    assert roi_from_box([0.4, 0.4, 0.6, 0.6], 0.5, 100, 100) == (30, 30, 70, 70)
    # clamps to the image and never collapses below 1px
    assert roi_from_box([0.0, 0.0, 0.05, 0.05], 1.0, 100, 100) == (0, 0, 10, 10)


@pytest.mark.unit
def test_map_box_to_full_is_inverse_of_cropping():
    roi = (30, 30, 70, 70)  # a 40x40 crop of a 100x100 image
    # a box centered in the crop maps back to the original box region
    assert map_box_to_full([0.25, 0.25, 0.75, 0.75], roi, 100, 100) == [0.4, 0.4, 0.6, 0.6]


@pytest.mark.unit
def test_autolabel_primed_maps_detection_back_to_full_frame():
    # detection sits centered in the ROI crop -> should land back on the priming box
    stub = StubLabeler([Detection("helicoil", BBox(0.25, 0.25, 0.75, 0.75), 0.9)])
    out = autolabel(_png(), "helicoil", prev_box=[0.4, 0.4, 0.6, 0.6], margin=0.5, labeler=stub)
    assert out["present"] is True and out["primed"] is True
    assert out["label"] == "helicoil"
    assert [round(v, 3) for v in out["box"]] == [0.4, 0.4, 0.6, 0.6]
    # it searched the cropped ROI (40x40), not the full frame
    assert stub.seen[0][0] == (40, 40)


@pytest.mark.unit
def test_autolabel_full_frame_when_no_prev_box():
    stub = StubLabeler([Detection("helicoil", BBox(0.1, 0.1, 0.3, 0.3), 0.8)])
    out = autolabel(_png(), "helicoil", prev_box=None, labeler=stub)
    assert out["present"] is True and out["primed"] is False
    assert [round(v, 3) for v in out["box"]] == [0.1, 0.1, 0.3, 0.3]
    assert stub.seen[0][0] == (100, 100)  # whole frame


@pytest.mark.unit
def test_autolabel_falls_back_to_full_frame_when_roi_empty():
    # ROI turns up nothing -> retry on the whole frame (object may have moved out of ROI)
    class RoiEmptyThenHit:
        def __init__(self):
            self.calls = 0

        def label(self, image, features):
            self.calls += 1
            if image.size != (100, 100):
                return []  # ROI crop: nothing
            return [Detection("helicoil", BBox(0.7, 0.7, 0.85, 0.85), 0.6)]

    lb = RoiEmptyThenHit()
    out = autolabel(_png(), "helicoil", prev_box=[0.1, 0.1, 0.2, 0.2], labeler=lb)
    assert lb.calls == 2  # ROI then full frame
    assert out["present"] is True and out["primed"] is False


@pytest.mark.unit
def test_autolabel_ignores_whole_frame_boxes_and_reports_absent():
    # open-vocab models sometimes return the entire image; that's not a localization
    stub = StubLabeler([Detection("helicoil", BBox(0.0, 0.0, 1.0, 1.0), 0.99)])
    out = autolabel(_png(), "helicoil", prev_box=None, labeler=stub)
    assert out["present"] is False and out["box"] is None
