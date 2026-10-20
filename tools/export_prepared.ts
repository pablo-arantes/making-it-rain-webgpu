/**
 * Prepare a system the way the browser does, and write it out for OpenMM.
 *
 * The plan names **cross-fed energies** as a Phase 3 gate, and it is the one
 * thing the term-by-term comparison does not cover. Every force in a prepared
 * `System` matches OpenMM's element for element on the parity fixtures, which
 * proves the parameters agree — it does not prove the two agree on an *energy*.
 * A wrong unit in `serialiseSystem`, or a convention mismatch in how a force is
 * written, would survive every test so far and show up here.
 *
 * It also covers the case that has no OpenMM counterpart at all: a system whose
 * hydrogens this code placed, which OpenMM never built and so cannot be
 * compared against structurally. Evaluating it in both engines is the only
 * check available for it.
 *
 *     npx tsx tools/export_prepared.ts fixtures/1L2Y/receptor.pdb fixtures/crossfeed
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { writePdb } from "../src/analysis/pdb.js";
import { readPdb } from "../src/analysis/pdb.js";
import { serialiseState } from "../src/engine/state.js";
import { NonbondedMethod, serialiseSystem } from "../src/engine/system.js";
import { totalCharge } from "../src/prep/assign.js";
import { inferBonds, parseStandardBonds } from "../src/prep/bonds.js";
import { mergeForceFields, parseForceField } from "../src/prep/ffxml.js";
import {
  addHydrogens,
  isHydrogenAtom,
  isWater,
  parseHydrogenDefinitions,
  stripAtoms,
} from "../src/prep/hydrogens.js";
import { prepareSystem } from "../src/prep/prepare.js";
import { solvate } from "../src/prep/solvate.js";
import { buildSystem } from "../src/prep/system.js";

const [input, outDir] = process.argv.slice(2);
if (input === undefined || outDir === undefined) {
  console.error("usage: export_prepared.ts <receptor.pdb> <out-dir>");
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });

const vendor = (name: string): string =>
  readFileSync(resolve("vendor", name), "utf8");

const field = mergeForceFields([
  parseForceField(vendor("ffxml/protein.ff14SB.xml")),
  parseForceField(vendor("ffxml/amber14-tip3p.xml")),
]);
const standardBonds = parseStandardBonds(vendor("ffxml/residues.xml"));
const definitions = parseHydrogenDefinitions(vendor("ffxml/hydrogens.xml"));
const waterBox = readPdb(vendor("water/tip3p.pdb"));

const deposited = readPdb(readFileSync(input, "utf8"));
const heavy = stripAtoms(stripAtoms(deposited, isWater).topology, isHydrogenAtom).topology;
const hydrogenated = addHydrogens(heavy, inferBonds(heavy, standardBonds), definitions, 7);
const solute = prepareSystem(hydrogenated.topology, field, standardBonds);
const solvated = solvate(
  hydrogenated.topology,
  hydrogenated.bonds,
  solute.assignment.atoms,
  waterBox,
  { padding: 0.4, model: "tip3p", ionicStrength: 0.15, neutralize: true, seed: 24301 },
);
const prepared = prepareSystem(solvated.topology, field, standardBonds);

// 🔴 `constraints: "none"`. Constrained bonds are removed from the
// HarmonicBondForce, so a constrained system and an unconstrained one have
// genuinely different bonded energies -- comparing one against the other would
// look like a bug in the bonded term. The engine applies constraints at run
// time; the energy comparison wants every term present.
const system = buildSystem(prepared, {
  nonbondedMethod: NonbondedMethod.PME,
  cutoff: 1,
  constraints: "none",
  removeCMMotion: false,
});

const box = solvated.boxVectors;
writeFileSync(resolve(outDir, "system.xml"), serialiseSystem(system));
writeFileSync(
  resolve(outDir, "state.xml"),
  serialiseState({
    openmmVersion: "8.4",
    stepCount: 0,
    time: 0,
    boxVectors: [
      [box[0], 0, 0],
      [0, box[1], 0],
      [0, 0, box[2]],
    ],
    particleCount: solvated.topology.atoms.length,
    positions: solvated.topology.positions,
  }),
);
writeFileSync(
  resolve(outDir, "system.pdb"),
  writePdb(solvated.topology, { remark: "PREPARED BY making-it-rain-webgpu" }),
);

console.log(
  `  ${solvated.topology.atoms.length} atoms, ${solvated.waters} waters, ` +
    `${solvated.positiveIons} Na+, ${solvated.negativeIons} Cl-, ` +
    `box ${box[0].toFixed(5)} nm, charge ${totalCharge(prepared.assignment).toFixed(6)} e`,
);
console.log(`  wrote system.xml, state.xml and system.pdb to ${outDir}`);
