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
@compute @workgroup_size(64)
fn wrap(@builtin(global_invocation_id) gid: vec3<u32>) {
  let atom = gid.x;
  if (atom >= params.particle_count) { return; }
  for (var axis = 0u; axis < 3u; axis++) {
    wrapped_positions[atom * 3u + axis] =
      wrap_axis(positions[atom * 3u + axis], params.box[axis].xy);
  }
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
  let x = axis_cell(atom, 0u);
  let y = axis_cell(atom, 1u);
  let z = axis_cell(atom, 2u);
  // Row-major in (x, y, z), matching the host builder and the pair kernel's
  // 27-cell scan.
  let cell = (x * params.divisions.y + y) * params.divisions.z + z;
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
