// The cell list, built on the device: a counting sort over the grid.
//
// It has to be here rather than on the host. Every step moves the atoms, so a
// host-built list means reading 26,749 positions back per step, and a readback
// is a full pipeline drain -- it would cost more than the forces.
//
// Four dispatches: clear the counts, assign each atom to a cell and count,
// scan the counts into offsets, scatter the atoms into place.
//
// 🔴 `scatter` claims slots with an atomic, so the order it produces *within* a
// cell depends on how the workgroups were scheduled. A fifth pass, `sort_cells`,
// replaces it with ascending atom index.
//
// That pass has now had three different justifications, and the history is the
// useful part.
//
// The first version argued the order could not matter, because the forces
// accumulate in 64-bit fixed point and an exact integer sum does not depend on
// the order its terms arrive in. Sound argument, false premise at the time:
// `nonbonded-pairs.wgsl` then summed one atom's contributions in an **f32
// register** and made a single fixed-point add at the end, so the per-atom sum
// reassociated with the neighbour order -- measured at 1.89e-3 kJ/mol/nm on the
// worst force component by feeding the same kernel two orderings of the same
// partition.
//
// 🔴 That premise is now true, and this comment claimed otherwise for longer
// than it was right. The pair kernels convert and add **every pair** in fixed
// point (`accumulate_pair`), so nothing reassociates and the order genuinely
// cannot change a force. The evidence is `validateNeighbours`: the cell-shell
// path combines 27 partial sums per atom and the neighbour-list path combines
// 8, which are different groupings of the same terms, and they agree on all
// 80,247 force components of 3HTB bit for bit. An inexact accumulator could
// not do that.
//
// So `sort_cells` stays for the two reasons that are still standing, neither of
// which is the force. It is what makes the device's list comparable to the host
// builder's, which is how the cell list is validated at all. And the neighbour
// list is built by walking this one, so sorting is what gives *that* list a
// defined order -- which AGENTS.md requires of every list a kernel walks, and
// which is cheap insurance against a future kernel that does sum in f32.

@group(0) @binding(0) var<storage, read>       positions: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> cell_of: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> starts: array<u32>;
@group(0) @binding(4) var<storage, read_write> order: array<u32>;
@group(0) @binding(5) var<uniform>             params: CellParams;
@group(0) @binding(6) var<storage, read_write> sorted: array<u32>;
@group(0) @binding(7) var<storage, read_write> wrapped_positions: array<vec2<u32>>;

struct CellParams {
  particle_count: u32,
  cell_count: u32,
  // How many cells along each axis.
  divisions: vec3<u32>,
  // The box, and one cell's extent along each axis, both in fixed point. Only
  // `.xy` is used -- low word then high.
  //
  // `vec4` rather than `vec2` because WGSL rounds an array element's stride up
  // to 16 bytes in the uniform address space, so an `array<vec2<u32>, 3>` is
  // laid out with eight bytes of padding after each element anyway. Writing the
  // padding into the type keeps the host's packing and the shader's view of it
  // from being two separate pieces of knowledge.
  box: array<vec4<u32>, 3>,
  cell_extent: array<vec4<u32>, 3>,
  /// 🔴 Appended, not inserted. The three off-diagonal lattice components in
  /// fixed point -- `b.x`, `c.x`, `c.y` -- and the inverse box in f32. Putting
  /// them before `box` would have moved every offset `packParams` writes, for
  /// no benefit; a uniform's layout is a contract with the host.
  ///
  /// Zero and diagonal respectively for a rectangular cell, which is how
  /// `skewed()` tells them apart.
  off: array<vec4<u32>, 3>,
  inverse: array<vec4<f32>, 3>,
}

/// One coordinate folded into [0, box), exactly.
///
/// Adding or subtracting whole boxes is exact, since `add64` and `sub64` are
/// integer operations, so the fold is exact however many boxes out the
/// coordinate started. Bounded rather than unbounded so a wild coordinate
/// cannot hang a dispatch.
fn wrap_axis(value: vec2<u32>, box: vec2<u32>) -> vec2<u32> {
  var folded = value;
  // A position may sit several boxes out: nothing wraps the coordinates during
  // a run, and OpenMM's own states do not either.
  for (var guard = 0u; guard < 64u; guard++) {
    if (!negative64(folded)) { break; }
    folded = add64(folded, box);
  }
  for (var guard = 0u; guard < 64u; guard++) {
    if (less64(folded, box)) { break; }
    folded = sub64(folded, box);
  }
  return folded;
}

/// Fold every coordinate into [0, box), for the pair kernel to difference.
///
/// 🔴 This pass exists to buy the pair kernel a *precondition*, not to save
/// work here. Both endpoints inside the box means a difference inside one box,
/// which means one conditional shift is the whole minimum image -- see
/// `nearest_image_box` in fixed64.wgsl. The general wrap it replaces cost
/// 74,662 us of a 3HTB step; folding costs three folds per atom per step
/// instead of three per candidate pair, which on 3HTB is 80,247 of them against
/// 268 million.
///
/// It shares `wrap_axis` with `axis_cell` deliberately. The pair kernel takes
/// its home cell from this list and its pair vectors from this buffer, and if
/// those two disagreed for an atom near a cell boundary it would scan a
/// neighbourhood it is not listed in -- atom i missing j while j still finds i,
/// which breaks Newton's third law and leaks momentum. One function means they
/// agree by construction rather than by two implementations matching.
/// Whether this cell is skewed, from the off-diagonals being non-zero.
///
/// A rectangular cell keeps the exact integer paths below unchanged, which is
/// what makes this conversion verifiable by "rectangular unchanged" rather
/// than by trusting that the general code reduces correctly.
fn skewed() -> bool {
  for (var slot = 0u; slot < 3u; slot++) {
    let value = params.off[slot].xy;
    if (value.x != 0u || value.y != 0u) { return true; }
  }
  return false;
}

fn lattice_diagonal() -> array<vec2<u32>, 3> {
  return array<vec2<u32>, 3>(
    params.box[0].xy, params.box[1].xy, params.box[2].xy,
  );
}

fn lattice_off() -> array<vec2<u32>, 3> {
  return array<vec2<u32>, 3>(
    params.off[0].xy, params.off[1].xy, params.off[2].xy,
  );
}

@compute @workgroup_size(64)
fn wrap(@builtin(global_invocation_id) gid: vec3<u32>) {
  let atom = gid.x;
  if (atom >= params.particle_count) { return; }
  if (skewed()) {
    // A skewed cell folds as a whole: subtracting lattice vector `c` moves the
    // `a` and `b` components too, so there is no per-axis version of this.
    let folded = fold_into_cell(
      array<vec2<u32>, 3>(
        positions[atom * 3u + 0u],
        positions[atom * 3u + 1u],
        positions[atom * 3u + 2u],
      ),
      lattice_diagonal(), lattice_off(),
      params.inverse[0].xyz, params.inverse[1].xyz, params.inverse[2].xyz,
    );
    for (var axis = 0u; axis < 3u; axis++) {
      wrapped_positions[atom * 3u + axis] = folded[axis];
    }
    return;
  }
  for (var axis = 0u; axis < 3u; axis++) {
    wrapped_positions[atom * 3u + axis] =
      wrap_axis(positions[atom * 3u + axis], params.box[axis].xy);
  }
}

/// Which cell an atom falls in, for a skewed cell, from its fractional
/// coordinate.
///
/// 🔴 f32, where the rectangular path below is exact integers, and the reason
/// that is safe is worth stating because this file already carries a warning
/// against recomputing a cell index in f32. That warning is about *two*
/// calculations disagreeing: the pair kernel used to recompute a home cell the
/// builder had bucketed differently, so atom i would scan a neighbourhood it
/// was not listed in. There is still exactly one authority here -- this
/// function -- and every reader takes the index from the list.
///
/// What an f32 ulp can do on its own is put an atom within ~5e-7 nm of a cell
/// boundary on either side of it. That is harmless: the 27-cell shell reaches a
/// whole cell width past the home cell, and a cell is at least a cutoff wide,
/// so the slack is the width minus the cutoff -- tenths of a nanometre, six
/// orders of magnitude more than the ulp. An exact version would need the
/// inverse box in fixed point and a signed 64-bit matrix multiply per atom.
fn skewed_cell(atom: u32) -> u32 {
  let x = from_fixed(wrapped_positions[atom * 3u + 0u]);
  let y = from_fixed(wrapped_positions[atom * 3u + 1u]);
  let z = from_fixed(wrapped_positions[atom * 3u + 2u]);
  var cell = vec3<u32>(0u, 0u, 0u);
  for (var axis = 0u; axis < 3u; axis++) {
    let row = vec3<f32>(
      params.inverse[0][axis], params.inverse[1][axis], params.inverse[2][axis],
    );
    let fraction = x * row.x + y * row.y + z * row.z;
    let divisions = params.divisions[axis];
    // `fold_into_cell` leaves this in [0, 1), but clamp anyway: an f32 estimate
    // at the upper edge must not index past the last cell.
    let scaled = i32(floor(fraction * f32(divisions)));
    cell[axis] = u32(clamp(scaled, 0, i32(divisions) - 1));
  }
  return (cell.x * params.divisions.y + cell.y) * params.divisions.z + cell.z;
}

/// Which cell an atom falls in along one axis.
///
/// Folds the coordinate, then counts how many cell extents fit. That loop runs
/// at most `divisions` times, which the cell list keeps small: a 6.5 nm box at
/// a 1 nm cutoff gives six.
fn axis_cell(atom: u32, axis: u32) -> u32 {
  let box = params.box[axis].xy;
  let wrapped = wrap_axis(positions[atom * 3u + axis], box);

  let divisions = params.divisions[axis];
  let extent = params.cell_extent[axis].xy;
  var edge = extent;
  var index = 0u;
  // `divisions - 1` is the last cell, and a coordinate at or past the final
  // edge belongs to it -- which is also how the host builder clamps.
  for (; index + 1u < divisions; index++) {
    if (less64(wrapped, edge)) { break; }
    edge = add64(edge, extent);
  }
  return index;
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x;
  if (cell >= params.cell_count) { return; }
  atomicStore(&counts[cell], 0u);
}

@compute @workgroup_size(64)
fn assign(@builtin(global_invocation_id) gid: vec3<u32>) {
  let atom = gid.x;
  if (atom >= params.particle_count) { return; }
  // Row-major in (x, y, z), matching the host builder and the pair kernel's
  // 27-cell scan. A skewed cell buckets from its fractional coordinate, which
  // `wrap` has already folded into [0, 1); a rectangular one keeps the exact
  // integer path.
  var cell: u32;
  if (skewed()) {
    cell = skewed_cell(atom);
  } else {
    let x = axis_cell(atom, 0u);
    let y = axis_cell(atom, 1u);
    let z = axis_cell(atom, 2u);
    cell = (x * params.divisions.y + y) * params.divisions.z + z;
  }
  cell_of[atom] = cell;
  atomicAdd(&counts[cell], 1u);
}

/// One workgroup, so the scan needs no second level and no device-wide barrier.
///
/// Each thread sums a contiguous chunk, thread 0 scans the 256 chunk totals
/// serially, then each thread writes its chunk's running offsets. 256 serial
/// adds once per step is nothing next to the forces, and it is much easier to
/// be sure of than a Hillis-Steele scan with its barriers. The cell count is
/// in the hundreds for these boxes and the chunking handles any size.
var<workgroup> chunk_offset: array<u32, 256>;

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_index) lid: u32) {
  let n = params.cell_count;
  let per_thread = (n + 255u) / 256u;
  let begin = min(lid * per_thread, n);
  let end = min(begin + per_thread, n);

  var total = 0u;
  for (var cell = begin; cell < end; cell++) {
    total = total + atomicLoad(&counts[cell]);
  }
  chunk_offset[lid] = total;
  workgroupBarrier();

  if (lid == 0u) {
    var running = 0u;
    for (var thread = 0u; thread < 256u; thread++) {
      let value = chunk_offset[thread];
      chunk_offset[thread] = running;
      running = running + value;
    }
  }
  workgroupBarrier();

  var running = chunk_offset[lid];
  for (var cell = begin; cell < end; cell++) {
    starts[cell] = running;
    let count = atomicLoad(&counts[cell]);
    running = running + count;
    // Leave the counts at zero so `scatter` can use them as its cursors, and
    // so the next step's `assign` starts from a clean slate.
    atomicStore(&counts[cell], 0u);
  }
  if (lid == 0u) {
    // Every atom lands in exactly one cell, so the final offset is the atom
    // count. The host reads this back once and checks it.
    starts[n] = params.particle_count;
  }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let atom = gid.x;
  if (atom >= params.particle_count) { return; }
  let cell = cell_of[atom];
  let slot = starts[cell] + atomicAdd(&counts[cell], 1u);
  order[slot] = atom;
}

/// Put each cell's atoms in ascending index order.
///
/// One thread per slot. The atom in that slot belongs at `start + rank`, where
/// `rank` counts the atoms of the same cell with a lower index -- found by
/// scanning the cell's own slice, which is the only part of the array that can
/// contain them. That is O(cell population) per atom, around 120 reads for
/// these boxes, and it is fully parallel: no atomics and no ordering between
/// threads, so the result is the same every time.
@compute @workgroup_size(64)
fn sort_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let slot = gid.x;
  if (slot >= params.particle_count) { return; }
  let atom = order[slot];
  let cell = cell_of[atom];
  let begin = starts[cell];
  let end = starts[cell + 1u];
  var rank = 0u;
  for (var other = begin; other < end; other++) {
    if (order[other] < atom) { rank = rank + 1u; }
  }
  sorted[begin + rank] = atom;
}
