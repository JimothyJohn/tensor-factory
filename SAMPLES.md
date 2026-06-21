# Sample prompts for helicoil generation

Prompts for `helicoils-synth --backend gemini` (Nano Banana / `gemini-2.5-flash-image`),
framed as a **quality inspection**: a person turning a worn, used machined part to judge
whether a wire thread insert is seated right in a tapped hole. Three hard-won rules:

- **The coil is wound TIGHT — it is not a spring.** The model renders "coiled wire insert"
  as a compression spring (gaps between turns, standing proud) unless you say *"wound tight
  with no gaps, forming continuous internal threads, not a loose spring."* A real installed
  Helicoil *becomes* the thread.
- **Shoot straight down into the bore.** The top-down / oblique angles render the coil as
  threads; grazing and shallow angles flatten it to a plain tapped hole or invite the
  spring look. Bias installed states to top-down.
- **Photoreal and worn, never shiny.** Clean studio renders look fake. Force grime, wear,
  uneven light, soft focus, and sensor grain.

> **Known limit (text-only):** a *correctly* installed Helicoil looks almost like a plain
> tapped hole, and the **proud** / **cross-threaded** states inherently show protruding
> coil — which the model still tends to draw as a spring. If you need those to look right,
> the lever is an **image reference** (Nano Banana accepts an input photo): pass a real
> installed-Helicoil macro and it conditions on the actual appearance.

Each prompt = an **inspection angle** + a **QC state** + a **worn neutral metal** + a
**finish/grime** hint + the **photoreal suffix**.

## Inspection angles

1. `looking straight down into a tapped hole, seeing deep into the bore`
2. `at an oblique three-quarter angle into a tapped hole`
3. `at a low grazing angle, looking almost flush along the top surface across the rim of a tapped hole`
4. `at a shallow angle into a tapped hole, the surface receding into focus`

## QC states

Every installed state carries the shared clause **`the wire wound tight with no gaps
between turns, forming continuous internal threads, not a loose spring`** (abbreviated
`<TIGHT>` below) and a preferred camera angle.

| State | Phrasing | Angle | Verdict | Renders well? |
|-------|----------|-------|---------|---------------|
| **flush / pass** | `correctly installed and seated flush, <TIGHT>, looking like a clean finely-threaded hole, nothing protruding` | top-down / oblique | good | ✅ |
| **slightly recessed** | `installed slightly too deep, <TIGHT>, the tight coil recessed a little below the rim` | top-down / oblique | fail (low) | ✅ |
| **missing** | `no insert installed, an empty tapped hole showing bare cut threads, rim dinged and dirty` | top-down / grazing | fail (absent) | ✅ |
| **damaged coil** | `a damaged thread, <TIGHT>, one coil loop nicked or deformed out of round down inside the bore` | top-down / oblique | fail (damaged) | ✅ |
| **slightly proud** | `seated slightly too high, <TIGHT>, the tight coil raised just a hair proud, still threaded into the bore` | grazing / oblique | fail (high) | ⚠️ spring-prone |
| **cross-threaded** | `installed crooked and cross-threaded, <TIGHT>, the coil seated visibly off-axis but still in the bore` | oblique / top-down | fail (crooked) | ⚠️ spring-prone |

Dropped from earlier iterations: **partial / backed-out** and **bridging tang** — both
reliably produced a spring standing on the surface.

Deliberately **no** "defective"/"failed"/"reject" words — those make the model render a
cartoonishly broken part. Describe only the physical seating; the verdict lives in the
manifest.

## Worn neutral metal (no color)

`dull worn aluminum` · `clear-anodized aluminum, matte and scuffed` ·
`bare stainless steel with a brushed finish` · `grey tool steel`. Clear-anodized at most —
never colored anodizing, brass, or coatings. Avoid the words *shiny*, *pristine*, *clean*.

## Finish / grime + lighting (push the noise)

- **grime:** `machining marks and faint oxidation` · `fine swarf chips and shop grime
  around the hole` · `dust, smudges, and a faint oil film` · `light corrosion, scuffs, and
  tiny scratches`
- **lighting:** `uneven shop lighting` · `a harsh single LED with glare and hard shadow` ·
  `dim angled microscope light with hotspots` · `soft directional light with deep shadows`

## Photoreal suffix (append to every prompt)

```
real macro photograph taken through an inspection microscope, natural uneven lighting,
fine sensor grain, shallow depth of field and slightly soft focus, used and worn, gritty
and imperfect, not a clean studio render
```

## Recipe

```
macro inspection photo <angle>, <state>, in <metal>, <grime>, <lighting>, <photoreal suffix>
```

### Levers for more variety

- **Seed sweep** — same prompt, different `--seed`: free variation in light, grime, and
  defect placement; also the cheapest way to re-roll a prompt that produced a loose spring.
- **Thread size** — `M3` / `M6` / `M10`, `coarse` / `fine thread` change coil pitch.
- **Magnification** — `filling most of the frame` vs `small in a wide field of metal`
  spreads box sizes (biggest lever for teaching localization).

## The 25 in `images/`

`images/` holds 25 inspection samples from this system (master seed `20260620`): 5 clean
passes and 4 each of recessed / proud / missing / cross-threaded / damaged. Filenames are
neutral (`sample_NN.png`) for **blind review**; the intended `qc_state`, `angle`, and full
prompt for each live in `images/manifest.json`. Reproduce any one:

```bash
uv run --with google-genai python -c "
from helicoils_synth.generator import NanoBananaGenerator
import json; m = json.load(open('images/manifest.json'))['samples'][10]
NanoBananaGenerator().generate(m['prompt'], m['seed']).image.save('repro.png')
"
```

## Generating a fresh batch

The batch tool is `packages/helicoils-synth/scripts/gen_samples.py` (this whole catalog
lives there as the single source of truth):

```bash
# Text-only, 25 samples into images/ (clears the old batch first):
uv run --with google-genai python packages/helicoils-synth/scripts/gen_samples.py

# Conditioned on a real photo -- the fix for proud / cross-threaded:
uv run --with google-genai python packages/helicoils-synth/scripts/gen_samples.py \
    --reference path/to/real_installed_helicoil.png
```

### Reference-conditioned generation

`NanoBananaGenerator.generate(..., reference=<PIL image>)` passes the photo as image input
so the model conditions on the real part's appearance instead of guessing. This is the
reliable way to get the states text alone renders as a spring. Drop one (or a few) real
macro photos of installed/proud/cross-threaded inserts and pass `--reference`; the prompt
then drives material/lighting/defect while the reference pins the *look* of a Helicoil.
