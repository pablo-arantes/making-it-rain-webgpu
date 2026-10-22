# Measurements

Machine-readable records produced by `tools/`, committed so a claim in `docs/`
can be traced to the run that produced it.  They are small JSON files; the
fixtures they describe (`system.xml`, `state.xml`, `system.pdb`) are **not**
committed -- regenerate them with `tools/build_reference_system.py`.

| File | Produced by | Machine |
| --- | --- | --- |
| `3HTB-JZ4-facts.json` | `tools/build_reference_system.py --pdb-id 3HTB --ligand JZ4` | Apple M3 Pro, OpenMM 8.4.0.dev, openmmforcefields 0.15.1, openff-toolkit 0.18.0, RDKit 2025.09.5 |
| `1L2Y-facts.json` | `tools/build_reference_system.py --pdb-id 1L2Y` | same |
| `restraint-mask.json` | `tools/probe_restraint_mask.py` | same |
| `host-timings.json` | `tools/bench_reference_host.py` | Apple M3 Pro (12 CPU cores, 14 GPU cores), 18 GB |
| `preparation-determinism.json` | `tools/probe_preparation_determinism.py` | same |
| `calibration-M3Pro-opencl.json` | `tools/calibrate_thresholds.py` | same — **provisional**: this host has no mixed-precision GPU platform, so the basis is OpenCL single. Replaced by a Colab T4 run |
| `device-M3Pro-chrome.json` | `npm run report:adapter` | Apple M3 Pro GPU through Chrome 155 headless, WebGPU/Metal |
| `profile-step-m3pro.json` | `web/validate.html` in Chrome, via `src/gpu/profile.ts` | same |
| `triclinic-dodecahedron-1L2Y.json` | `tools/build_reference_system.py --box-shape dodecahedron` | OpenMM 7.7.0.dev — **superseded**, see below |
| `triclinic-reference-parity.json` | `tools/build_triclinic_fixture.py` | OpenMM 8.4.0.dev (`parametrizani_env`), Apple M3 Pro |
| `triclinic-gpu-dodecahedron.json` | `npm run report:verify` and `web/validate.html` | Apple M3 Pro GPU through Chrome |
| `bench-M3Pro-chrome.json` | `npm run bench` | same |
