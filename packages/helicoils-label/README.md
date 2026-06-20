# helicoils-label

Label Studio integration for the helicoil annotation loop: push auto-label candidates in
as pre-annotations, correct them by hand, pull clean labels back out as COCO.

```bash
# Start Label Studio (separate install: `uv pip install label-studio`)
label-studio start --port 8080

# Push a synth/FLUX dataset (COCO + GroundingDINO predictions) as a labeling project.
# Images must be reachable by the LS server -- serve them and pass --image-base:
python -m http.server 8081 --directory data/ &
helicoils-label --token "$LABEL_STUDIO_API_KEY" push --data data/ \
  --title "helicoil v1" --image-base http://localhost:8081

# ... correct the boxes in the browser ...

# Pull the corrected annotations back to COCO for training:
helicoils-label --token "$LABEL_STUDIO_API_KEY" pull --project 3 --out data/labeled.coco.json
helicoils-train fit --data data/ --out model.onnx   # (after wiring labeled COCO)
```

`helicoils-label config` prints the labeling-config XML. URL/token also read from
`LABEL_STUDIO_URL` / `LABEL_STUDIO_API_KEY`.

Licensed under Apache-2.0.
