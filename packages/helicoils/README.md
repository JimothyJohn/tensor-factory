# helicoils

Open, lightweight helicoil detection. This core package is dependency-free and CPU-only:
the canonical bounding box, the 8-bit detection codec, and annotation-format conversions
(COCO / YOLO / Pascal VOC).

```python
from helicoils import BBox, encode_uint8, decode_uint8, max_error_px

box = BBox(0.1, 0.1, 0.9, 0.9)        # normalized xyxy, top-left origin
packed = encode_uint8(box)            # (26, 26, 230, 230) -- four uint8
again = decode_uint8(packed)          # round-trips within ~1 px at 480
max_error_px(480)                     # 0.94 -- inside the 3 px budget
```

Licensed under Apache-2.0.
