// Real-space Ewald and Lennard-Jones over the pairs inside the cutoff.
//
// Gathering: an invocation sums the interactions of one atom, so the pair loop
// needs no atomics for its own force.  Every pair is therefore evaluated twice,
// once from each end, and the energy is halved to compensate.
//
// Two ways of finding the candidates, and `accumulate_pair` is the one copy of
// what happens once a candidate is inside the cutoff.
//
//   `direct_pairs_list`  walks a Verlet neighbour list, which is the fast path
//                        and what a run uses.
//   `direct_pairs`       walks the 27 cells around the atom's own, which needs
//                        no list and is the fallback for an atom whose list
//                        overflowed.  It is also the primary path in the
//                        differential tests, so the two are compared against
//                        each other as well as against the reference.
//
// `build_neighbours` is the list builder, and it lives here because finding
// candidates in the 27-cell shell is exactly what `direct_pairs` does.
//
// Exclusions are skipped, not zeroed. An excluded pair's reciprocal-space
// contribution is removed, and its own interaction applied, by the companion
// kernel in nonbonded-exceptions.wgsl -- together those are most of what
// OpenMM reports for this force group. See src/reference/forces/nonbonded.ts.

@group(0) @binding(3) var<storage, read> particle: array<vec4<f32>>;  // q, sigma, eps, _
@group(0) @binding(4) var<storage, read> lists: array<u32>;
@group(0) @binding(5) var<uniform>       params: Params;

/// The Verlet neighbour list, and how much of it each atom uses.
///
/// 🔴 Indexed `[k * particle_count + atom]`, not `[atom * capacity + k]`. The
/// pair kernel's threads are consecutive atoms at the same `k`, so this layout
/// makes one coalesced read where the obvious one makes 64 scattered ones.
///
/// `count` is `OVERFLOWED` for an atom with more neighbours than the list has
/// room for. Those atoms are skipped here and picked up by `direct_pairs`,
/// because dropping them would silently lose every pair they are in.
@group(0) @binding(6) var<storage, read_write> neighbours: array<u32>;
@group(0) @binding(7) var<storage, read_write> neighbour_count: array<u32>;
/// `[0]` is the rebuild flag, `[2]` records an overflow. See verlet.wgsl.
@group(0) @binding(8) var<storage, read_write> verlet_flags: array<atomic<u32>>;

const OVERFLOWED: u32 = 0xffffffffu;

struct Params {
  particle_count: u32,
  /// Where the sorted atom order begins in `lists`; the cell starts precede it.
  sorted_offset: u32,
  /// Where each atom's exclusion-list start index begins in `lists`.
  exclusion_start_offset: u32,
  /// Where the flat exclusion list begins in `lists`.
  exclusion_list_offset: u32,
  /// Where each atom's cell index begins in `lists`.
  cell_of_offset: u32,
  /// How many neighbours the list has room for per atom.
  verlet_capacity: u32,
  /// How many invocations share one atom's list; see `direct_pairs_list`.
  verlet_slices: u32,
  /// `(cutoff + skin)^2`, the radius the list is built to.
  verlet_cutoff_squared: f32,
  /// Three `u32` of padding were here: `cells` is a `vec3<u32>`, which WGSL
  /// aligns to 16 bytes, so the three fields above cost nothing and
  /// `PAIR_PARAMS_BYTES` is unchanged.
  cells: vec3<u32>,
  cutoff: f32,
  box: vec3<f32>,
  alpha: f32,
  /// The box and half the box per axis, in fixed point, for the exact minimum
  /// image. `vec4` because WGSL gives a uniform array element a 16-byte
  /// stride; only `.xy` is used, low word then high.
  box_fixed: array<vec4<u32>, 3>,
  half_box_fixed: array<vec4<u32>, 3>,
  /// The three off-diagonal lattice components in fixed point -- `b.x`, `c.x`,
  /// `c.y` -- and the inverse box in f32. Zero and diagonal for a rectangular
  /// cell, which is what `frame.skewed` is computed from.
  off_fixed: array<vec4<u32>, 3>,
  inverse: array<vec4<f32>, 3>,
}

/// The box and half the box, in fixed point, held in registers.
///
/// 🔴 Hoisted out of the pair loop deliberately. Reading them from the uniform
/// inside `pair_vector` meant six `vec4` loads per candidate pair, and a bulk
/// water atom has a few thousand candidates: it cost 23% of the throughput on
/// 1L2Y and 34% on 3HTB.
///
/// Hoisting *this atom's own position* the same way was tried and measured
/// neutral, so it is not done. `delta_image_axis` re-reads `positions[a * 3 +
/// axis]` for every candidate, which in a gather kernel is loop-invariant, and
/// the same argument that worked for the box says it should pay. It does not:
/// 1L2Y 1,152 -> 1,123 us and 3HTB 437,782 -> 437,792 us, both inside the
/// 0.1% the repeated measurements vary by. Threads in a workgroup are
/// consecutive atoms in the same cell, so these loads hit cache; the box loads
/// were from a uniform, which is a different path. The argument was not wrong
/// about the redundancy, only about it costing anything.
struct FixedBox {
  box: array<vec2<u32>, 3>,
  half: array<vec2<u32>, 3>,
  off: array<vec2<u32>, 3>,
  inverse0: vec3<f32>,
  inverse1: vec3<f32>,
  inverse2: vec3<f32>,
  /// 🔴 Decided once per invocation, outside the candidate loop. A skewed cell
  /// needs the general reduction, which costs an f32 matrix product and three
  /// `from_fixed` per candidate; a rectangular one keeps the single conditional
  /// shift it had. Measured on 1L2Y and 3HTB after this branch went in: the
  /// rectangular path did not move, which is what the branch is for.
  skewed: bool,
}

fn load_fixed_box() -> FixedBox {
  var loaded: FixedBox;
  loaded.box[0] = params.box_fixed[0].xy;
  loaded.box[1] = params.box_fixed[1].xy;
  loaded.box[2] = params.box_fixed[2].xy;
  loaded.half[0] = params.half_box_fixed[0].xy;
  loaded.half[1] = params.half_box_fixed[1].xy;
  loaded.half[2] = params.half_box_fixed[2].xy;
  loaded.off[0] = params.off_fixed[0].xy;
  loaded.off[1] = params.off_fixed[1].xy;
  loaded.off[2] = params.off_fixed[2].xy;
  loaded.inverse0 = params.inverse[0].xyz;
  loaded.inverse1 = params.inverse[1].xyz;
  loaded.inverse2 = params.inverse[2].xyz;
  loaded.skewed =
    (loaded.off[0].x | loaded.off[0].y |
     loaded.off[1].x | loaded.off[1].y |
     loaded.off[2].x | loaded.off[2].y) != 0u;
  return loaded;
}

/// The pair vector, nearest image, wrapped in fixed point before conversion.
///
/// 🔴 Reads *wrapped* positions -- binding 0 of this kernel is the `wrap` pass's
/// output, not the integrator's buffer -- which is what lets this use
/// `nearest_image_box` and its single shift. See that function, and the `wrap`
/// entry point in cell-list.wgsl.
///
/// Deliberately not three `delta_image_axis` calls: the box would come from the
/// uniform each time. The arithmetic is identical -- difference in fixed point,
/// wrap in fixed point, convert last -- so the result is bit for bit the same.
fn pair_vector(a: u32, b: u32, frame: ptr<function, FixedBox>) -> vec3<f32> {
  let dx = sub64(positions[b * 3u + 0u], positions[a * 3u + 0u]);
  let dy = sub64(positions[b * 3u + 1u], positions[a * 3u + 1u]);
  let dz = sub64(positions[b * 3u + 2u], positions[a * 3u + 2u]);
  if ((*frame).skewed) {
    return nearest_image_triclinic(
      array<vec2<u32>, 3>(dx, dy, dz),
      (*frame).box, (*frame).off,
      (*frame).inverse0, (*frame).inverse1, (*frame).inverse2,
    );
  }
  return vec3<f32>(
    nearest_image_box(dx, (*frame).box[0], (*frame).half[0]),
    nearest_image_box(dy, (*frame).box[1], (*frame).half[1]),
    nearest_image_box(dz, (*frame).box[2], (*frame).half[2]),
  );
}

/// Decompose a flat cell index back into its three axis indices.
///
/// Row-major in (x, y, z), matching `cell-list.wgsl` and the host builder.
fn cell_axes(cell: u32, cells: vec3<u32>) -> vec3<u32> {
  let z = cell % cells.z;
  let rest = cell / cells.z;
  return vec3<u32>(rest / cells.y, rest % cells.y, z);
}

/// Does `other` appear in `atom`'s exclusion list?
///
/// Checked only after the distance test, because almost every candidate fails
/// that first: a 1.0 nm sphere at water density holds about 240 atoms out of
/// the few thousand the 27 cells contain.
fn is_excluded(atom: u32, other: u32) -> bool {
  let start = lists[params.exclusion_start_offset + atom];
  let end = lists[params.exclusion_start_offset + atom + 1u];
  for (var k = start; k < end; k++) {
    if (lists[params.exclusion_list_offset + k] == other) { return true; }
  }
  return false;
}

/// One pair, from atom `i`'s end, added into this invocation's accumulators.
///
/// 🔴 The two candidate loops share this rather than each carrying a copy,
/// because the arithmetic has to be identical to the bit: the energy takes
/// *two* separate `to_fixed` additions, Coulomb and then Lennard-Jones, and
/// folding them into one `to_fixed(coulomb + lj)` is a different number. That
/// is also why the accumulators come in by pointer instead of being returned --
/// a returned f32 would have to be added somewhere, and where decides the
/// rounding.
fn accumulate_pair(
  me: vec4<f32>,
  other: vec4<f32>,
  d: vec3<f32>,
  r_squared: f32,
  energy_sum: ptr<function, vec2<u32>>,
  force_x: ptr<function, vec2<u32>>,
  force_y: ptr<function, vec2<u32>>,
  force_z: ptr<function, vec2<u32>>,
) {
  let r = sqrt(r_squared);
  let alpha_r = params.alpha * r;
  let charge_product = 138.93545764498457 * me.x * other.x;
  let screened = erfc_approx(alpha_r);
  // Halved here rather than at the end: each pair is seen from both ends, and
  // scaling by 0.5 is exact in f32, so this is the same total and keeps the
  // running sum a pure integer.
  *energy_sum = add64(*energy_sum, to_fixed(0.5 * charge_product * screened / r));
  var d_e_d_r = charge_product *
    (-screened / r_squared - params.alpha * erf_derivative(alpha_r) / r);

  let epsilon = sqrt(me.z * other.z);
  if (epsilon != 0.0) {
    let sigma = 0.5 * (me.y + other.y);
    let s2 = sigma * sigma / r_squared;
    let s6 = s2 * s2 * s2;
    let s12 = s6 * s6;
    *energy_sum = add64(*energy_sum, to_fixed(2.0 * epsilon * (s12 - s6)));
    d_e_d_r += (4.0 * epsilon / r) * (6.0 * s6 - 12.0 * s12);
  }
  // F_i = (dE/dr / r) * d, and this invocation owns only atom i.
  let contribution = (d_e_d_r / r) * d;
  *force_x = add64(*force_x, to_fixed(contribution.x));
  *force_y = add64(*force_y, to_fixed(contribution.y));
  *force_z = add64(*force_z, to_fixed(contribution.z));
}

/// The 27-cell shell scan.
///
/// The fallback, and the reference path. A run reaches it only for an atom
/// whose neighbour list overflowed, which the gate below selects; the
/// differential tests dispatch it for every atom by building no list at all, so
/// `neighbour_count` is zero everywhere and the gate is inverted. Both are one
/// comparison outside the loop.
@compute @workgroup_size(64)
fn direct_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particle_count) { return; }
  // Atoms the list covers are handled by `direct_pairs_list`.
  if (params.verlet_capacity != 0u && neighbour_count[i] != OVERFLOWED) { return; }

  // Which of the 27 cells around the home cell this invocation handles.
  //
  // 🔴 One invocation per atom left only `particle_count` threads in flight --
  // 2373 on 1L2Y, which is 38 workgroups of 64 and nowhere near enough to hide
  // memory latency on a GPU that wants tens of thousands. Splitting the
  // neighbour shell across the dispatch's y axis multiplies the thread count by
  // 27 without changing the work. It is exact: the per-atom force already
  // accumulates through `accumulate_into`, whose fixed-point atomics are
  // order-independent by construction, so 27 partial sums land on the same
  // total as one. See the measurement in docs/measurements/.
  let shell = gid.y;
  let dx = i32(shell / 9u) - 1;
  let dy = i32((shell / 3u) % 3u) - 1;
  let dz = i32(shell % 3u) - 1;

  let me = particle[i];
  let cutoff_squared = params.cutoff * params.cutoff;
  let cells = params.cells;
  var frame = load_fixed_box();

  // Which cell this atom is in, read from the list rather than recomputed.
  //
  // 🔴 It used to be recomputed here, from the wrapped fractional coordinate in
  // f32. That is a different calculation from the one that built the list --
  // which buckets in fixed point -- and at the 6.5 nm edge of the 3HTB box an
  // f32 coordinate has a 4.8e-7 nm ulp, so the two could disagree for an atom
  // near a cell boundary. In a gather kernel that is worse than it sounds: atom
  // i would scan the neighbourhood of a cell it is not listed in, so i could
  // miss j while j still finds i. The pair contributes to one atom's force and
  // not the other's, which breaks Newton's third law and leaks momentum. Taking
  // the home cell from the list makes the two agree by construction.
  let home = cell_axes(lists[params.cell_of_offset + i], cells);

  // Accumulated in 64-bit fixed point, in registers, rather than in an f32
  // vector.
  //
  // An atom in bulk water has around 240 neighbours inside a 1 nm cutoff and
  // its net force is a difference of large opposing contributions, so an f32
  // running sum loses significance to cancellation. Integers do not, and the
  // cost is a `to_fixed` and a carry per component per pair.
  //
  // What it bought, measured: the energy-drift gate on 1L2Y went from 0.376 to
  // 0.0428 kJ/mol per atom per ns, against OpenMM's own 0.374 on the same
  // configuration -- so most of what was being measured as drift was noise in
  // the *reported* energy, which is summed the same way and over the same 240
  // terms.
  //
  // What it did not buy, equally measured: 3HTB's direct force relative RMS
  // stayed at 1.13e-5, thirteen times 1L2Y's. So cancellation in the per-atom
  // sum is not what made that fixture worse, and an earlier version of this
  // comment said it was. The cause was the pair vector, now fixed:
  // `pair_vector` wraps in fixed point before converting, where the old
  // `delta_periodic` converted first and lost 4.8e-7 nm at the edge of a 6.5 nm
  // box. That took 3HTB to 5.37e-7 -- a factor of 21, and better than 1L2Y.
  // docs/NONBONDED.md has the whole episode, including why the bound estimated
  // from the pair-vector error was an order of magnitude too small.
  var force_x = vec2<u32>(0u, 0u);
  var force_y = vec2<u32>(0u, 0u);
  var force_z = vec2<u32>(0u, 0u);
  var energy_sum = vec2<u32>(0u, 0u);

  let neighbour = (vec3<i32>(home) + vec3<i32>(dx, dy, dz) + vec3<i32>(cells))
                % vec3<i32>(cells);
  let cell = (u32(neighbour.x) * cells.y + u32(neighbour.y)) * cells.z
           + u32(neighbour.z);
  let start = lists[cell];
  let end = lists[cell + 1u];
  for (var slot = start; slot < end; slot++) {
    let j = lists[params.sorted_offset + slot];
    if (j == i) { continue; }

    let d = pair_vector(i, j, &frame);
    let r_squared = dot(d, d);
    if (r_squared >= cutoff_squared) { continue; }
    if (is_excluded(i, j)) { continue; }

    accumulate_pair(
      me, particle[j], d, r_squared,
      &energy_sum, &force_x, &force_y, &force_z,
    );
  }

  accumulate_into(i * 3u + 0u, force_x);
  accumulate_into(i * 3u + 1u, force_y);
  accumulate_into(i * 3u + 2u, force_z);
  // Each pair was seen from both ends; the halving happened as each
  // contribution was formed.
  accumulate_energy_fixed(energy_sum);
}

/// The same forces, over a Verlet neighbour list instead of the cell shell.
///
/// 🔴 Why this exists, measured on 3HTB at 26,749 atoms: a 27-cell shell is 27
/// cutoff volumes around a sphere of 4.19, so only 13.7% of the 3,344
/// candidates an atom tests are inside the cutoff. A list built to
/// `cutoff + skin` holds about 520 of them and stays valid for as long as no
/// atom has moved half the skin, so the shell scan is paid once every few dozen
/// steps instead of every step.
///
/// One atom's list is split across `verlet_slices` invocations, striding rather
/// than taking contiguous blocks. Striding is what keeps the reads coalesced:
/// at a given `k` the invocations in a workgroup are consecutive atoms, and the
/// list is stored `[k * particle_count + atom]`, so they read neighbouring
/// words. It also keeps the slices balanced whatever the counts are. The split
/// is for occupancy, the same reason the shell is split across the dispatch --
/// one invocation per atom is 2,373 threads on 1L2Y -- and it is exact for the
/// same reason: the accumulators are fixed-point atomics.
@compute @workgroup_size(64)
fn direct_pairs_list(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particle_count) { return; }
  let count = neighbour_count[i];
  // Overflowed atoms belong to `direct_pairs`, which has no capacity to run out
  // of.
  if (count == OVERFLOWED) { return; }

  let me = particle[i];
  let cutoff_squared = params.cutoff * params.cutoff;
  var frame = load_fixed_box();

  var force_x = vec2<u32>(0u, 0u);
  var force_y = vec2<u32>(0u, 0u);
  var force_z = vec2<u32>(0u, 0u);
  var energy_sum = vec2<u32>(0u, 0u);

  let n = params.particle_count;
  for (var k = gid.y; k < count; k += params.verlet_slices) {
    let j = neighbours[k * n + i];

    // The list is built to `cutoff + skin`, so the cutoff still has to be
    // tested here -- and the exclusions are tested after it, because most
    // candidates fail this.
    let d = pair_vector(i, j, &frame);
    let r_squared = dot(d, d);
    if (r_squared >= cutoff_squared) { continue; }
    if (is_excluded(i, j)) { continue; }

    accumulate_pair(
      me, particle[j], d, r_squared,
      &energy_sum, &force_x, &force_y, &force_z,
    );
  }

  accumulate_into(i * 3u + 0u, force_x);
  accumulate_into(i * 3u + 1u, force_y);
  accumulate_into(i * 3u + 2u, force_z);
  accumulate_energy_fixed(energy_sum);
}

/// Building the neighbour list: count per shell cell, prefix-sum, then fill.
///
/// 🔴 Three passes rather than one, and the reason is the two things profiling
/// found in `direct_pairs`. A single pass would be one invocation per atom
/// looping the 27 cells -- which is `particle_count` threads and a triple
/// nested loop, the exact pair of defects that made the pair kernel 18x slower
/// than it needed to be. Measured, on 1L2Y: one build that way cost 26.7 ms,
/// against 0.84 ms for the whole shell-scan force kernel, which does strictly
/// more arithmetic.
///
/// Splitting it per (atom, shell cell) fixes both -- 27x the threads, and the
/// shell loop becomes a flat one -- but then an atomic append would put the
/// list in whatever order the scheduler produced. AGENTS.md requires every list
/// a kernel walks to be in a defined order, so instead each cell's count is
/// taken first, prefix-summed over the 27 cells, and the fill writes at the
/// offset that gives it. The order is shell cell then slot within the cell, the
/// same order the one-invocation version produced, whatever the scheduler does.
/// The price is scanning the shell twice, which is still a small fraction of
/// what one pass cost.
@group(0) @binding(9) var<storage, read_write> shell_counts: array<u32>;

/// Decode this invocation's shell cell, and find its range in the sorted order.
///
/// Returns `vec2(start, end)`, empty when the flag says no rebuild is wanted.
fn shell_range(atom: u32, shell: u32) -> vec2<u32> {
  let cells = params.cells;
  let home = cell_axes(lists[params.cell_of_offset + atom], cells);
  let offset = vec3<i32>(i32(shell / 9u) - 1, i32((shell / 3u) % 3u) - 1, i32(shell % 3u) - 1);
  let neighbour = (vec3<i32>(home) + offset + vec3<i32>(cells)) % vec3<i32>(cells);
  let cell = (u32(neighbour.x) * cells.y + u32(neighbour.y)) * cells.z
           + u32(neighbour.z);
  return vec2<u32>(lists[cell], lists[cell + 1u]);
}

/// How many of one shell cell's atoms are inside `cutoff + skin` of this atom.
@compute @workgroup_size(64)
fn count_neighbours(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (atomicLoad(&verlet_flags[0]) == 0u) { return; }
  let i = gid.x;
  if (i >= params.particle_count) { return; }

  var frame = load_fixed_box();
  let range = shell_range(i, gid.y);
  var count = 0u;
  for (var slot = range.x; slot < range.y; slot++) {
    let j = lists[params.sorted_offset + slot];
    if (j == i) { continue; }
    let d = pair_vector(i, j, &frame);
    if (dot(d, d) >= params.verlet_cutoff_squared) { continue; }
    count += 1u;
  }
  shell_counts[gid.y * params.particle_count + i] = count;
}

/// Turn the 27 per-cell counts into the offsets the fill writes at.
///
/// Exclusive prefix sum, in place, 27 terms in one invocation -- small enough
/// that a parallel scan would cost more in dispatches than it saved.
@compute @workgroup_size(64)
fn offset_neighbours(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (atomicLoad(&verlet_flags[0]) == 0u) { return; }
  let i = gid.x;
  let n = params.particle_count;
  if (i >= n) { return; }

  var total = 0u;
  for (var shell = 0u; shell < 27u; shell++) {
    let here = shell_counts[shell * n + i];
    shell_counts[shell * n + i] = total;
    total += here;
  }
  // 🔴 Record the overflow and mark the atom, rather than truncating quietly.
  // `direct_pairs` then evaluates it from the cell shell in full. Losing a pair
  // is a discontinuity in the force, not a rounding error.
  if (total > params.verlet_capacity) {
    atomicOr(&verlet_flags[2], 1u);
    neighbour_count[i] = OVERFLOWED;
    return;
  }
  neighbour_count[i] = total;
}

/// Write one shell cell's neighbours into the atom's list.
@compute @workgroup_size(64)
fn fill_neighbours(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (atomicLoad(&verlet_flags[0]) == 0u) { return; }
  let i = gid.x;
  let n = params.particle_count;
  if (i >= n) { return; }
  // An atom whose list overflowed has no room to write into, and the cell-shell
  // path owns it.
  if (neighbour_count[i] == OVERFLOWED) { return; }

  var frame = load_fixed_box();
  let range = shell_range(i, gid.y);
  var at = shell_counts[gid.y * n + i];
  for (var slot = range.x; slot < range.y; slot++) {
    let j = lists[params.sorted_offset + slot];
    if (j == i) { continue; }
    let d = pair_vector(i, j, &frame);
    if (dot(d, d) >= params.verlet_cutoff_squared) { continue; }
    neighbours[at * n + i] = j;
    at += 1u;
  }
}
