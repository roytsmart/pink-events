# pink-events

An investigation of the many small, dim, pink events visible in the dark
interiors of supergranule cells in the ESIS level-4 inversions.

## The observation

While preparing the SPD 2026 presentation
([spd-2026](https://github.com/roytsmart/spd-2026)), the full-field RGB
movies of the MART-inverted O V 630 Å line (2019 September 30,
18:06–18:11 UTC) showed numerous small, faint, pink features inside the
supergranular cell interiors, away from the bright network that hosts
the well-known explosive events.

## The first question: what does pink mean?

The RGB movies map wavelength to color with `colorsynth`, which mixes
colors additively over the spectral axis. Pink is therefore ambiguous,
and the two readings are physically very different:

1. **Violet**: the extreme blue end of the visible mapping. A dim pink
   event would be a small, faint feature blueshifted by roughly
   100–150 km/s, a fast upflow.
2. **Magenta**: red and blue present with no green between them. A dim
   pink event would be a bidirectional profile, both wings and no line
   center, the classic explosive-event signature at small scale.

These look nearly the same in an additive rendering, so the first task
is to pull the recovered spectral profiles at a sample of the events and
see which they are: one displaced component, or two.

## Questions after that

- Are they real? MART is semiconvergent and the cell interiors are the
  low-signal regime, so the events must be shown to survive changes in
  iteration count and to appear consistently in overlapping channels
  rather than being reconstruction artifacts.
- How do they relate to the magnetic network? The claim "inside the
  cells" should be made against the HMI magnetograms rather than by eye.
- Do they persist, recur, or move over the ~5 minutes of the flight?
- How do their sizes, brightnesses, and velocities compare to the
  network explosive events (e.g. event E of the SPD work)?
- Is there a counterpart population in IRIS Si IV rasters?

## Resources

- `spd2026` (the [spd-2026](https://github.com/roytsmart/spd-2026)
  package) has the machinery this work starts from: cached level-4
  inversions, the RGB rendering, event cropping and tracking helpers,
  and AIA/HMI context loading, mostly in `spd2026.figures._level_4`.
- `esis.data.Level_4` for the inversions themselves.
- `sdo.hmi.open()` / `sdo.aia.open()` for context imagery.

Note that ESIS level-4 coordinates moved by one pixel when the WCS
origin was fixed across the group's packages in August 2026 (see
sun-data/interface-region-imaging-spectrograph#45,
sun-data/solar-dynamics-observatory#20 and #21); anything positional
derived before that fix is suspect at the half-arcsecond level.
