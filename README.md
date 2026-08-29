# Zephrock

Generalized Hoek-Brown → equivalent Mohr-Coulomb rock-mass strength calculator
— a free, from-scratch alternative to the relevant core of Rocscience's
RSData / RocData / RocLab. Part of the zeph-toolkit (Zephmatic ~ DIPS,
Zephslide ~ Slide, Zephflac ~ FLAC).

**Scope:** core generalized Hoek-Brown (Hoek, Carranza-Torres & Corkum, 2002)
→ equivalent Mohr-Coulomb (mb, s, a; rock-mass UCS/tensile/global strength;
sigma3max for slopes or tunnels; equivalent c'/phi'; a rough deformation
modulus estimate). Not RSData's other failure criteria.

## Quick start — browser GUI

Double-click `launch_app.bat` (Windows). First run installs dependencies
automatically, then opens the app in your browser. Later runs skip straight
to launch.

Manual equivalent:
```
pip install -r requirements.txt
streamlit run app.py
```

### What the GUI does

- **Results tab** — enter sigci, mi, GSI, D and a sigma3max convention
  (slope, tunnel, or a custom value); see mb/s/a, rock-mass strength values,
  and the equivalent c'/phi', with a downloadable text summary.
- **Envelope chart tab** — plots the real Hoek-Brown curve against the fitted
  Mohr-Coulomb line over your chosen sigma3max range.
- **Plug into Zephslide tab** — hands the computed material straight to
  Zephslide's circular-search solver for a real bench (height, face angle,
  unit weight) and reports the Bishop factor of safety, with no manual
  re-typing of c'/phi'.
- **About / validation tab** — the validation story and a documented,
  honest limitation (this tool could not exactly reproduce one project's
  own literature-reported RocLab output, because that source never states
  its sigma3max/depth assumption — see the tab for the full explanation).

## Quick start — library / scripts

```python
from zephrock import HoekBrownMaterial

hb = HoekBrownMaterial(sigci=69.04, mi=9, GSI=70)
sigma3max = hb.sigma3max_slope(unit_weight_MNm3=0.0265, height_m=10.0)
mc = hb.equivalent_mohr_coulomb(sigma3max)
print(mc.text_summary())

# Hand off directly into Zephslide/Zephflac:
material = hb.to_material(sigma3max=sigma3max, unit_weight_kNm3=26.5)
```

Zephrock's core (`zephrock/hoek_brown.py`) only needs `numpy` — it does not
import zephslide/zephflac unless you call `to_material()` /
`to_mohr_coulomb_material()`. This deployment bundles `zeph_common` and
`zephslide` too, so the GUI's "Plug into Zephslide" tab works out of the box.

## Package layout

```
app.py                 Streamlit GUI (this deployment's entry point)
requirements.txt
launch_app.bat
zephrock/
    __init__.py
    hoek_brown.py       core module — usable completely standalone
zephslide/              bundled so the "Plug into Zephslide" tab works
zeph_common/            shared Bench dataclass
```
