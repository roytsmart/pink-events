# pink-events

An investigation of the many small, dim, pink events visible in the dark
interiors of supergranule cells in one IRIS raster.

## The observation

While preparing the SPD 2026 presentation
([spd-2026](https://github.com/roytsmart/spd-2026)), the RGB rendering
of the IRIS Si IV 1394 Å raster shown on the opening slides had numerous
small, faint, pink features inside the supergranular cell interiors,
away from the bright network that hosts the well-known explosive events.

The raster is OBSID **3881013496**, *Very large dense raster 131.7x175
400s Si IV Mg II h/k Deep x 30*:

- 2013 October 22, 11:30:30–15:04:33 UT
- 400 slit positions, roughly 32 s cadence, 30 s exposures
- 139 x 182 arcsec field centered near (310, -95) arcsec
- spectral windows C II 1336, Si IV 1394, Mg II k 2796

## The first question: what does pink mean?

The RGB rendering maps wavelength to color with `colorsynth`, which
mixes colors additively over the spectral axis. Pink is therefore
ambiguous, and the two readings are physically very different:

1. **Violet**: the extreme blue end of the visible mapping. A dim pink
   event would be a small, faint feature blueshifted by roughly
   100-150 km/s, a fast upflow.
2. **Magenta**: red and blue present with no green between them. A dim
   pink event would be a bidirectional profile, both wings and no line
   center, the classic explosive-event signature at small scale.

These look nearly the same in an additive rendering, but IRIS measures
the spectra directly, so the distinction is one plot away: pull the
Si IV profiles at a sample of the pink pixels and see whether they are
one displaced component or two.

## Questions after that

- Are they above the background? The cell interiors are the faintest
  Si IV regime, and a 30 s deep exposure still has real noise and
  fiducial/pedestal artifacts there.
- How do they relate to the magnetic network? The claim "inside the
  cells" should be made against a magnetogram or the Mg II wing images
  rather than by eye.
- The raster sweeps each position once, so evolution is not directly
  observable, but the 3.5 hour, 400-step sweep gives statistics: how
  many are there, and what are their sizes, brightnesses, and
  velocities compared to the network explosive events?
- Do C II 1336 and Mg II k 2796 see them too? Both windows are in the
  same files.

## Resources

- The raster is already cached at
  `~/.iris/cache/iris_l2_20131022_113030_3881013496_raster/`.
- `iris.sg.SpectrographObservation.from_fits` loads it;
  `radiance` converts to physical units; the RGB rendering came from
  `named_arrays.plt.rgbmesh`.
- Beware `colorsynth.rgb()` on the full raster: it transiently
  allocates about 17x the size of its input
  (sun-data/colorsynth#10).
