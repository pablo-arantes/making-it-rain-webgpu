# Preparation: from a PDB to a System

Phase 3's job is to build, in the browser, the same `System` the notebook
builds in Python. The engine contract is OpenMM `System` XML, so "the same" is
checkable rather than rhetorical: build one here, serialise one there, compare
them field by field.

This file records what that comparison currently says, and what is not built
yet. Every number is produced by `test/prep-*.test.ts` against fixtures from
`tools/dump_parameters.py` and `tools/dump_template_match.py`.

## What runs

`web/prepare.html` drives all of this: a PDB ID or an upload in, and
`system.xml`, `state.xml` and `system.pdb` out, with the deviations that applied
listed on the page.

```
PDB text
  -> readPdb              src/analysis/pdb.ts
  -> stripAtoms           src/prep/hydrogens.ts  drop water and any hydrogens
  -> addHydrogens         src/prep/hydrogens.ts  variants, then the missing H
  -> solvate              src/prep/solvate.ts    water box, clashes, ions
  -> inferBonds           src/prep/bonds.ts      standard + disulfide + CONECT
  -> matchResidues        src/prep/template.ts   graph matching, not by name
  -> assignAtoms          src/prep/assign.ts     mass, charge, sigma, epsilon
  -> findAngles/Propers/Impropers + assign*      src/prep/bonded.ts
  -> assignImpropers      src/prep/torsions.ts   ordering="amber"
  -> findExceptions       src/prep/exceptions.ts 1-2, 1-3, scaled 1-4
  -> buildSystem          src/prep/system.ts     an OpenMMSystem the engine runs
  -> serialiseSystem      src/engine/system.ts   back out as OpenMM System XML
```

Writing `System` XML was the missing half of the contract: it could be read but
not produced, which is fine while every system comes from the notebook and not
fine once the browser makes one, because the user then has no artefact.
`test/system-serialise.test.ts` takes four files OpenMM wrote, writes each back
out, and requires the re-parse to be identical — OpenMM's file is the fixed
point, since a writer and reader wrong in the same way would round-trip
perfectly.

`prepareSystem` in `src/prep/prepare.ts` is the whole chain in one call.

## Parity, measured

Both fixtures are proteins with explicit hydrogens and no solvent, built by
`tools/build_prep_fixture.py`. Against OpenMM 8.4 with
`amber14/protein.ff14SB.xml`, `nonbondedMethod=NoCutoff`, `constraints=None`:

| | 1L2Y | 3HTB receptor | 1L2Y solvated |
| --- | --- | --- | --- |
| atoms | 304 | 2614 | 2373 |
| residues | 20 | 163 | 713 |
| bonds | 310 | 2634 | 1686 |
| angles | 565 | 4751 | 1253 |
| torsions | 1017 | 8842 | 1017 |
| exceptions | 1687 | 14232 | 3751 |
| template choice and atom correspondence | identical | identical | identical |
| masses, charges, sigma, epsilon | identical | identical | identical |
| every force in the built `System` | identical, in index order | identical, in index order | identical, in index order |
| unmatched terms | 0 | 0 | 0 |
| ambiguous parameter rows | 0 | 0 | 0 |

Hydrogen addition is compared separately, rebuilding from the deposited
receptor with water and all hydrogens removed — the same starting point OpenMM
is given. The variant chosen per residue, the full atom list in order, and every
bond match atom for atom on both proteins (1L2Y 154→304 atoms, 3HTB 1301→2614),
and the `System` built from *our* hydrogens is identical to the one OpenMM built
from *its own*, since parameters do not depend on where a hydrogen sits. That
closes the path from a deposited PDB to a runnable system. Positions are
deviation 16; the unreachable HID/HIE rule is deviation 17.

Solvation is compared the way the algorithm splits. Water placement is exactly
reproducible, so it is compared position by position: for 1L2Y at the
notebook's 4 Å padding and TIP3P, the box comes out 3.17632 nm against OpenMM's
3.17632, with **699 waters in the same places to 4.4e-16 nm** and 2401 atoms.
Ion placement is not reproducible in OpenMM (deviation 18), so only the counts
are compared, and they are exact: 2 Na⁺ and 3 Cl⁻, 694 waters, 2391 atoms.

The end of the chain is checked too: the solvated topology matches templates for
`HOH`, `NA` and `CL`, comes out neutral to round-off, and builds a `System` with
a PME nonbonded force and rigid water — a deposited PDB to a runnable system.

The third column is the notebook's actual configuration — 1L2Y in 688 TIP3P
waters with 2 Na+ and 3 Cl-, under `protein.ff14SB.xml` plus
`amber14/tip3p.xml`. It is the only fixture that exercises `mergeForceFields`,
the `HOH` template and the ion templates.

"Identical, in index order" means `toEqual` on the typed arrays with no
tolerance. That is affordable because both sides read the same decimal strings
out of the same ffxml and do the same arithmetic on them.

3HTB is there because 1L2Y cannot test what it does not contain: it is a
20-residue peptide with no histidine, no cysteine and no sulfur. T4 lysozyme has
all twenty standard residues, and its histidine comes out as **HID** — HID and
HIE share an element signature and differ only in which ring nitrogen carries
the proton, so that is the graph matcher choosing a protonation state.

## The four findings worth keeping

**Templates are matched by graph, not by name.** A PDB says `ALA` for every
alanine; ff14SB has `ALA`, `NALA` and `CALA`. A name-based matcher gets 25 of
1L2Y's 304 charges wrong, worst 0.5958 e. ff14SB has twelve element-signature
groups holding two templates each, and 1L2Y hits three of them: ILE/LEU three
times, decided *only* by the bond graph since both are mid-chain with identical
external bond counts, plus TYR/CPHE and SER/CALA, decided by the external bond
count.

**Charge comes from the residue template, not the atom type.** ff14SB's
`<NonbondedForce>` carries `<UseAttributeFromResidue name="charge"/>`, and the
same type takes different charges in different residues — a backbone `N` is
-0.4157 in `ALA` and -0.5163 in `NALA`. Reading charge from the type table
flattens them and leaves the total charge wrong.

**🔴 OpenMM caches improper matches by type signature and replays positions.**
`ordering="amber"` breaks improper ties on residue index and *template* atom
index, but `impr_cache` stores the positional permutation its rules chose and
reuses it for every later candidate with the same type signature. So the rules
effectively run once per signature, and the first candidate to carry it decides
the atom order for all the rest. On 1L2Y this changes two of the 1017 torsions:
tyrosine's `(49, 45, 50, 53)` caches positions `(1, 3, 0, 2)` and the later
`(51, 47, 52, 53)` would freshly compute `(3, 1, 0, 2)`. Swapping a torsion's
first two atoms flips the sign of its dihedral angle, so it is an energy
difference. A test computes the uncached answer and asserts it still differs in
exactly two torsions.

Two hypotheses were wrong before that one was right, and both were killed by
measurement rather than argument: that OpenMM's failure to re-read elements
between the ordering swaps was responsible (the asymmetry is real, but fixing it
changes none of the 1017), and that tyrosine's symmetric ring admitted a second
valid isomorphism that we had picked (the template indices are identical to
OpenMM's for all 304 atoms).

**Bond order is part of the contract.** A `HarmonicBondForce` holds bonds in the
order they were added. Every other force in the built `System` matched element
for element on the first run; the bonds differed in index order only, because
`inferBonds` sorted. It now follows `Topology.createStandardBonds` — chain by
chain, residue by residue, the order the bonds appear in `residues.xml`, each in
the direction the table names it, then disulfides, then CONECT. Order carries no
physics; it carries comparability.

**Two unit conversions that are one ulp apart.** OpenMM converts ångström to
nanometres by *multiplying by 0.1*; `30.806 / 10` and `30.806 * 0.1` are
different doubles, 4.4e-16 nm apart. That is far below anything physical and
exactly enough to fail an exact box comparison for a reason that looks like
nothing at all. `readPdb` multiplies, so the two agree bit for bit and the
`System` comparison can run with no tolerance. Relatedly, `createSystem` sets
the box whenever the topology has one — even with `NoCutoff` — so `buildSystem`
must not tie the box to whether the method is periodic.

A smaller one: a removed exception gets `sigma = 1.0`, not the pair's mean. It
cannot reach the energy because epsilon is zero, but it is what the serialised
`System` holds, and it differs on 875 of 1L2Y's 1687 exceptions.

## Not built yet

* **Force fields beyond ff14SB.** ff99SB and ff99SB-ILDN are vendored and parse;
  neither has been run through a parity comparison.
* **DHFR** as a third fixture, which the plan's gate names.
* **`prmtop`/`inpcrd` export**, and the protein analyses (Cα RMSD, Rg, RMSF, 2D
  RMSD, PCA, cross-correlation).
* **Cross-fed energies** — prepare here, evaluate in OpenMM, and the reverse.
  The `System` comparison is stronger term by term, but it does not prove the
  two agree on an energy, and that is the gate the plan asks for.
