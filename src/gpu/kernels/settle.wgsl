// SETTLE: the analytic solution for a rigid three-atom cluster, one thread each.
//
// Transcribed from src/reference/settle.ts, which is itself transcribed from
// `ReferenceSETTLEAlgorithm.cpp`. Analytic, so there is no iteration and no
// tolerance: the result satisfies all three distances to the precision of the
// arithmetic.
//
// Two entry points. `settle_cluster` projects the trial positions;
// `settle_velocities` removes the velocity component along each of the three
// bonds, by the same analytic route -- a 3x3 solve for the three constraint
// multipliers, not an iteration.
//
// The velocity solve is here rather than left to the iterative kernel because
// the reference uses it, and leaving it out was not the harmless approximation
// it looked like. With the rigid waters sent through SHAKE instead, a
// thermostatted step came out 2.36 nm/ps from the reference -- the whole
// off-manifold velocity, because the device's cluster tables hold only the
// *iterative* clusters and so 688 of 1L2Y's 784 got no velocity projection at
// all. It hid from the end-to-end test: the position solve and the velocity
// correction that follows it reconstruct most of what a bad RATTLE got wrong,
// which is why it only showed once the thermostat broke that cancellation and
// once the step was compared stage by stage.
//
// Everything is computed in f32 *relative to the reference positions*. The
// displacements a step produces are around 1e-4 nm and the reference bond
// vectors around 0.1 nm, so f32 has room for both; the absolute coordinates,
// which reach 6.5 nm, stay in fixed point and are never converted. See
// docs/PRECISION.md.

@group(0) @binding(0) var<storage, read>       reference_position: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> trial: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read>       mass: array<f32>;
@group(0) @binding(3) var<storage, read>       cluster_atoms: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read>       cluster_distances: array<vec2<f32>>;
@group(0) @binding(5) var<uniform>             params: Params;
@group(0) @binding(6) var<storage, read_write> velocities: array<f32>;

struct Params {
  cluster_count: u32,
}

/// The reference position of one atom, relative to another's: exact in fixed
/// point, then converted.
fn reference_delta(atom_a: u32, atom_b: u32) -> vec3<f32> {
  // Not `from`/`to` and not `origin`/`target`: WGSL reserves a long list of
  // words for future use, and `from` and `target` are both on it.
  return vec3<f32>(
    from_fixed(sub64(reference_position[atom_b * 3u + 0u], reference_position[atom_a * 3u + 0u])),
    from_fixed(sub64(reference_position[atom_b * 3u + 1u], reference_position[atom_a * 3u + 1u])),
    from_fixed(sub64(reference_position[atom_b * 3u + 2u], reference_position[atom_a * 3u + 2u])),
  );
}

/// How far an atom's trial position sits from its reference position.
fn displacement(atom: u32) -> vec3<f32> {
  return vec3<f32>(
    from_fixed(sub64(trial[atom * 3u + 0u], reference_position[atom * 3u + 0u])),
    from_fixed(sub64(trial[atom * 3u + 1u], reference_position[atom * 3u + 1u])),
    from_fixed(sub64(trial[atom * 3u + 2u], reference_position[atom * 3u + 2u])),
  );
}

/// Write a trial position back as reference + displacement, in fixed point.
fn store(atom: u32, offset: vec3<f32>) {
  for (var axis = 0u; axis < 3u; axis++) {
    let slot = atom * 3u + axis;
    var component = offset.x;
    if (axis == 1u) { component = offset.y; }
    if (axis == 2u) { component = offset.z; }
    trial[slot] = add64(reference_position[slot], to_fixed(component));
  }
}

@compute @workgroup_size(64)
fn settle_cluster(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.cluster_count) { return; }
  let atoms = cluster_atoms[index];
  let a0 = atoms.x;   // the central atom, whose two bonds are equal
  let a1 = atoms.y;
  let a2 = atoms.z;
  let distance1 = cluster_distances[index].x;
  let distance2 = cluster_distances[index].y;

  let m0 = mass[a0];
  let m1 = mass[a1];
  let m2 = mass[a2];
  let xp0 = displacement(a0);
  let xp1 = displacement(a1);
  let xp2 = displacement(a2);
  let b0 = reference_delta(a0, a1);
  let c0 = reference_delta(a0, a2);

  let invTotalMass = 1.0 / (m0 + m1 + m2);
  let com = (xp0 * m0 + (b0 + xp1) * m1 + (c0 + xp2) * m2) * invTotalMass;

  let a1v = xp0 - com;
  let b1v = b0 + xp1 - com;
  let c1v = c0 + xp2 - com;

  // An orthonormal frame: z along the reference triangle's normal, x in the
  // plane containing the central atom's displaced position.
  let aksZd = cross(b0, c0);
  let aksXd = cross(a1v, aksZd);
  let aksYd = cross(aksZd, aksXd);
  let ex = aksXd / length(aksXd);
  let ey = aksYd / length(aksYd);
  let ez = aksZd / length(aksZd);

  let xb0d = dot(ex, b0);
  let yb0d = dot(ey, b0);
  let xc0d = dot(ex, c0);
  let yc0d = dot(ey, c0);
  let za1d = dot(ez, a1v);
  let xb1d = dot(ex, b1v);
  let yb1d = dot(ey, b1v);
  let zb1d = dot(ez, b1v);
  let xc1d = dot(ex, c1v);
  let yc1d = dot(ey, c1v);
  let zc1d = dot(ez, c1v);

  // The canonical rigid triangle in this frame.
  let rc = 0.5 * distance2;
  var rb = sqrt(distance1 * distance1 - rc * rc);
  let ra = rb * (m1 + m2) * invTotalMass;
  rb = rb - ra;
  let sinphi = za1d / ra;
  let cosphi = sqrt(1.0 - sinphi * sinphi);
  let sinpsi = (zb1d - zc1d) / (2.0 * rc * cosphi);
  let cospsi = sqrt(1.0 - sinpsi * sinpsi);

  let ya2d = ra * cosphi;
  var xb2d = -rc * cospsi;
  let yb2d = -rb * cosphi - rc * sinpsi * sinphi;
  let yc2d = -rb * cosphi + rc * sinpsi * sinphi;
  let xb2d2 = xb2d * xb2d;
  let hh2 = 4.0 * xb2d2 + (yb2d - yc2d) * (yb2d - yc2d) + (zb1d - zc1d) * (zb1d - zc1d);
  let deltx = 2.0 * xb2d + sqrt(4.0 * xb2d2 - hh2 + distance2 * distance2);
  xb2d = xb2d - deltx * 0.5;

  // The rotation about z that best matches the trial positions.
  let alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d;
  let beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d;
  let gamma = xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d;
  let al2be2 = alpha * alpha + beta * beta;
  let sintheta = (alpha * gamma - beta * sqrt(al2be2 - gamma * gamma)) / al2be2;
  let costheta = sqrt(1.0 - sintheta * sintheta);

  let a3d = vec3<f32>(-ya2d * sintheta, ya2d * costheta, za1d);
  let b3d = vec3<f32>(xb2d * costheta - yb2d * sintheta,
                      xb2d * sintheta + yb2d * costheta, zb1d);
  let c3d = vec3<f32>(-xb2d * costheta - yc2d * sintheta,
                      -xb2d * sintheta + yc2d * costheta, zc1d);

  // Back to the laboratory frame. The forward transform took components along
  // ex, ey and ez -- `dot(ex, v)` and friends -- so the way back is to rebuild
  // the vector from those components, not to apply the transpose.
  let a3 = ex * a3d.x + ey * a3d.y + ez * a3d.z;
  let b3 = ex * b3d.x + ey * b3d.y + ez * b3d.z;
  let c3 = ex * c3d.x + ey * c3d.y + ez * c3d.z;

  store(a0, com + a3);
  store(a1, com + b3 - b0);
  store(a2, com + c3 - c0);
}

/// A particle's velocity.
fn velocity_of(atom: u32) -> vec3<f32> {
  return vec3<f32>(
    velocities[atom * 3u + 0u],
    velocities[atom * 3u + 1u],
    velocities[atom * 3u + 2u],
  );
}

/// RATTLE for a rigid triangle, solved rather than iterated.
///
/// Transcribed from `settleVelocities` in src/reference/settle.ts, itself from
/// `ReferenceSETTLEAlgorithm::applyToVelocities`. The three multipliers `tab`,
/// `tbc` and `tca` are the solution of the 3x3 system that makes all three
/// relative velocities along the bonds vanish at once; applying them along the
/// bond directions, scaled by inverse mass, is the correction.
@compute @workgroup_size(64)
fn settle_velocities(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.cluster_count) { return; }
  let atoms = cluster_atoms[index];
  let a = atoms.x;
  let b = atoms.y;
  let c = atoms.z;

  let m_a = mass[a];
  let m_b = mass[b];
  let m_c = mass[c];
  if (m_a == 0.0 || m_b == 0.0 || m_c == 0.0) { return; }
  let inv_a = 1.0 / m_a;
  let inv_b = 1.0 / m_b;
  let inv_c = 1.0 / m_c;

  // Bond directions, from the reference positions these velocities belong to.
  let e_ab = normalize(reference_delta(a, b));
  let e_bc = normalize(reference_delta(b, c));
  let e_ca = normalize(reference_delta(c, a));

  let v_a = velocity_of(a);
  let v_b = velocity_of(b);
  let v_c = velocity_of(c);
  // The relative velocity along each bond, which the correction must cancel.
  let v_ab = dot(v_b - v_a, e_ab);
  let v_bc = dot(v_c - v_b, e_bc);
  let v_ca = dot(v_a - v_c, e_ca);

  // Cosines of the triangle's interior angles, and their sines squared.
  let c_a = -dot(e_ab, e_ca);
  let c_b = -dot(e_ab, e_bc);
  let c_c = -dot(e_bc, e_ca);
  let s2_a = 1.0 - c_a * c_a;
  let s2_b = 1.0 - c_b * c_b;
  let s2_c = 1.0 - c_c * c_c;

  let mabc_inv = 1.0 / (m_a * m_b * m_c);
  let denom =
    (((s2_a * m_b + s2_b * m_a) * m_c +
      (s2_a * m_b * m_b + 2.0 * (c_a * c_b * c_c + 1.0) * m_a * m_b + s2_b * m_a * m_a)) *
       m_c +
     s2_c * m_a * m_b * (m_a + m_b)) * mabc_inv;
  let total = m_a + m_b + m_c;
  let t_ab =
    ((c_b * c_c * m_a - c_a * m_b - c_a * m_c) * v_ca +
     (c_a * c_c * m_b - c_b * m_c - c_b * m_a) * v_bc +
     (s2_c * m_a * m_a * m_b * m_b * mabc_inv + total) * v_ab) / denom;
  let t_bc =
    ((c_a * c_b * m_c - c_c * m_b - c_c * m_a) * v_ca +
     (s2_a * m_b * m_b * m_c * m_c * mabc_inv + total) * v_bc +
     (c_a * c_c * m_b - c_b * m_a - c_b * m_c) * v_ab) / denom;
  let t_ca =
    ((s2_b * m_a * m_a * m_c * m_c * mabc_inv + total) * v_ca +
     (c_a * c_b * m_c - c_c * m_b - c_c * m_a) * v_bc +
     (c_b * c_c * m_a - c_a * m_b - c_a * m_c) * v_ab) / denom;

  let new_a = v_a + (e_ab * t_ab - e_ca * t_ca) * inv_a;
  let new_b = v_b + (e_bc * t_bc - e_ab * t_ab) * inv_b;
  let new_c = v_c + (e_ca * t_ca - e_bc * t_bc) * inv_c;
  for (var axis = 0u; axis < 3u; axis++) {
    velocities[a * 3u + axis] = component_of(new_a, axis);
    velocities[b * 3u + axis] = component_of(new_b, axis);
    velocities[c * 3u + axis] = component_of(new_c, axis);
  }
}

/// Pull one component out of a vec3 by a runtime index.
fn component_of(value: vec3<f32>, axis: u32) -> f32 {
  if (axis == 0u) { return value.x; }
  if (axis == 1u) { return value.y; }
  return value.z;
}
