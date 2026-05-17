/*******************************************************************************
    Copyright (c) The Taichi MPM Authors (2018- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/geometry/mesh.h>
#include <taichi/visualization/pakua.h>
#include "mpm.h"
#include "particles.h"
#include "boundary_particle.h"
#include "rigid_body_solver.h"
#include <taichi/system/profiler.h>

TC_NAMESPACE_BEGIN

// added: helpers to convert a generic Vector to the engine's per-dim torque
// type when applying external torque to a rigid body. In 3D the torque is a
// 3-vector (same as Vector). In 2D the torque is a scalar, and we adopt the
// convention that the scalar is stored in component [0] of the user-supplied
// Vector. These helpers let advect_rigid_bodies() write a single non-branched
// call to apply_torque() that compiles for both dims.
template <int dim>
struct AppliedTorqueAdapter {};

template <>
struct AppliedTorqueAdapter<2> {
  // For 2D: AngularVelocity<2>::ValueType == real (scalar).
  static real to_engine_torque(const VectorND<2, real> &v) { return v[0]; }
};

template <>
struct AppliedTorqueAdapter<3> {
  // For 3D: AngularVelocity<3>::ValueType == Vector3 (== VectorND<3, real>).
  static VectorND<3, real> to_engine_torque(const VectorND<3, real> &v) {
    return v;
  }
};

void check_scripting_parameters(const Config &config) {
  TC_ASSERT_INFO(!config.has_key("scripted"),
                 "'scripted' is deprecated. Please remove."); ///////////// ????
  TC_ASSERT_INFO(!config.has_key("position"),
                 "Use 'initial_position' instead of 'position'.")
  TC_ASSERT_INFO(!config.has_key("rotation"),
                 "Use 'initial_rotation' instead of 'rotation'.")

  if (config.has_key("scripted_position")) {
    TC_ASSERT_INFO(!config.has_key("initial_position"),
                   "scripted_position and initial_position cannot coexist.");
    TC_ASSERT_INFO(!config.has_key("initial_velocity"),
                   "scripted_position and initial_velocity cannot coexist.");
  } else {
    TC_ASSERT_INFO(config.has_key("initial_position"),
                   "Please specify one (and only one) of 'scripted_position' "
                   "and 'initial_position'.");
  }

  if (config.has_key("scripted_rotation")) {
    TC_ASSERT_INFO(!config.has_key("initial_rotation"),
                   "scripted_rotation and initial_rotation cannot coexist!");
    TC_ASSERT_INFO(
        !config.has_key("initial_angular_velocity"),
        "scripted_rotation and initial_angular_velocity cannot coexist!");
  }
  if (config.has_key("friction")) {
    TC_ASSERT_INFO(!config.has_key("friction0"),
                   "friction and friction0 cannot coexist!");
    TC_ASSERT_INFO(!config.has_key("friction1"),
                   "friction and friction1 cannot coexist!");
  }
  if (config.has_key("friction0") != config.has_key("friction1")) {
    TC_ERROR("friction0 and friction1 must be specified simultaneuously.");
  }

  if (config.has_key("friction0")) {
    TC_ASSERT_INFO(!config.has_key("friction"),
                   "friction0 and friction cannot coexist.");
  }
}

// create rigid body, called from this file ------------------------------------
template <int dim>
std::unique_ptr<RigidBody<dim>> MPM<dim>::create_rigid_body(Config config) {
  std::unique_ptr<RigidBody<dim>> rigid_ptr =
      std::make_unique<RigidBody<dim>>();
  auto &rigid = *rigid_ptr;
  rigid.velocity = config.get("initial_velocity", Vector(0.0f));
  rigid.angular_velocity =
      config.get("initial_angular_velocity",
                 typename AngularVelocity<dim>::ValueType(0.0_f));
  check_scripting_parameters(config);
  if (config.has_key("friction")) {
    rigid.frictions[0] = rigid.frictions[1] = config.get("friction", 0.0f);
  } else {
    rigid.frictions[0] = config.get("friction0", 0.0f);
    rigid.frictions[1] = config.get("friction1", 0.0f);
  }
  rigid.restitution = config.get("restitution", 0.0f);
  rigid.codimensional = config.get<bool>("codimensional");
  rigid.color = config.get("color", Vector3(0.5_f));
  rigid.linear_damping = config.get("linear_damping", 0.0f);
  rigid.angular_damping = config.get("angular_damping", 0.0f);

  // added: external (driving) force and torque on the rigid body's COM.
  // Two ways to specify each:
  //   STATIC:    config["applied_force"]    = Vector(...)
  //              config["applied_torque"]   = Vector(...)
  //   FUNCTION:  config["applied_force_fn"] = uint64 pointer to Function13
  //              config["applied_torque_fn"]= uint64 pointer to Function13
  // The function form takes precedence if both are given. The function is
  // called every substep with the current simulation time; it lets you
  // gate the drive, ramp it, or schedule arbitrary time-varying loads
  // without re-creating the rigid body.
  rigid.applied_force  = config.get("applied_force",  Vector(0.0_f));
  rigid.applied_torque = config.get("applied_torque", Vector(0.0_f));

  // Read optional function-form drive, exactly the same way as
  // scripted_position / scripted_rotation are read above.
  typename RigidBody<dim>::ForceFunctionType *force_fn =
      (typename RigidBody<dim>::ForceFunctionType *)config.get(
          "applied_force_fn", (uint64)0);
  typename RigidBody<dim>::TorqueFunctionType *torque_fn =
      (typename RigidBody<dim>::TorqueFunctionType *)config.get(
          "applied_torque_fn", (uint64)0);
  if (force_fn) {
    rigid.applied_force_func = *force_fn;
    rigid.applied_force_func_id = config.get<int>("applied_force_fn_id");
  }
  if (torque_fn) {
    rigid.applied_torque_func = *torque_fn;
    rigid.applied_torque_func_id = config.get<int>("applied_torque_fn_id");
  }

  // added: "freeze on axis exit" runtime safety net (for dynamic bodies).
  // Opt-in via freeze_on_axis_exit=True in the Python config.
  rigid.freeze_on_axis_exit =
      config.get("freeze_on_axis_exit", false);
  rigid.freeze_axis     = config.get("freeze_axis",     0);
  rigid.freeze_axis_min = config.get("freeze_axis_min", -1e30_f);
  rigid.freeze_axis_max = config.get("freeze_axis_max",  1e30_f);
  rigid.is_frozen       = false;

  typename RigidBody<dim>::PositionFunctionType *f =
      (typename RigidBody<dim>::PositionFunctionType *)config.get(
          "scripted_position", (uint64)0);
  typename RigidBody<dim>::RotationFunctionType *g =
      (typename RigidBody<dim>::RotationFunctionType *)config.get(
          "scripted_rotation", (uint64)0);

  // if there is scripted pos (f) ----------------------------------------------
  if (f) {
    rigid.pos_func = *f;
    rigid.pos_func_id = config.get<int>("scripted_position_id");
  }
  // if there is scripted rot (g) ----------------------------------------------
  if (g) {
    rigid.rot_func = *g;
    rigid.rot_func_id = config.get<int>("scripted_rotation_id");
  }

  MatrixP mesh_to_world(1.0_f);

  // initial values ------------------------------------------------------------
  Vector initial_position;
  // if there is scripted pos (f)
  if (f) {
    initial_position = (*f)(this->current_t);
  // else
  } else {
    initial_position = config.get<Vector>("initial_position");
  }
  rigid.position = initial_position;

  // rotation value ------------------------------------------------------------
  // 2d
  TC_STATIC_IF(dim == 2) {
    real angle = id(0);
    if (g) {
      angle = (*g)(id(this->current_t));
    } else {
      angle = config.get("initial_rotation", 0.0_f);
    }
    rigid.rotation = id(Rotation<2>(radians(angle)));
  }

  // 3d ----------
  TC_STATIC_ELSE {
    Vector euler;
    // if there is scripted rot (g)
    if (rigid.rot_func) {
      euler = rigid.rot_func(id(this->current_t));
    // else
    } else {
      euler = config.get("initial_rotation", id(Vector(0.0_f)));
    }
    // quaternion : rotation value (q)
    euler = radians(id(euler));
    Eigen::Quaternion<real> q =
        Eigen::AngleAxis<real>(euler[0], Eigen::Matrix<real, 3, 1>::UnitX()) *
        Eigen::AngleAxis<real>(euler[1], Eigen::Matrix<real, 3, 1>::UnitY()) *
        Eigen::AngleAxis<real>(euler[2], Eigen::Matrix<real, 3, 1>::UnitZ());
    rigid.rotation.value = id(q);
  }
  TC_STATIC_END_IF

  // rotation axis -------------------------------------------------------------
  if (dim == 3) {
    rigid.rotation_axis = config.get<Vector>("rotation_axis", Vector(0.0_f));
  }

  return rigid_ptr;
}

// add rigid particle, called from mpm.cpp -------------------------------------
template <int dim>
void MPM<dim>::add_rigid_particle(Config config) {

  // create rigid body
  auto rigid_ptr = create_rigid_body(config);
  auto &rigid = *rigid_ptr;

  // This config is for particles
  // particle type
  Config config_new = config;
  config_new.set("rigid", &rigid);
  // rigid particle density
  real density;
  if (config.get<bool>("codimensional")) {
    density = config.get("density", 40.0f);
  } else {
    density = config.get("density", 400.0f);
  }

  // add boundary particle, called from this file ------------------------------
  std::vector<ParticlePtr> added_particles;
  auto add_boundry_particle = [&](
    Vector position, Vector normal,
    typename RigidBody<dim>::ElementType *untransformed) {
    config_new.set("normal", normal)
        .set("rigid", &rigid)
        .set("offset", position - rigid.position);
    auto alloc = allocator.allocate_particle("rigid_boundary");
    RigidBoundaryParticle<dim> *p =
        static_cast<RigidBoundaryParticle<dim> *>(alloc.second);
    p->pos = position;
    p->initialize(config_new);
    p->untransformed_element = *untransformed;
    added_particles.push_back(alloc.first);
  };

  // mesh ----------------------------------------------------------------------
  rigid.mesh = std::make_unique<typename RigidBody<dim>::MeshType>();
  auto &mesh = rigid.mesh;
  mesh->initialize(config);
  Vector center_of_mass;
  // Initialize position and rotation
  Vector scale = config.get<Vector>("scale", Vector(1.0_f));
  Vector initial_position = rigid.position;
  // First, scale to make sure we have the correct mass and inertia
  auto &elements = rigid.mesh->elements;
  for (auto &elem : elements) {
    for (int k = 0; k < dim; k++) {
      elem.v[k] = scale * elem.v[k];
    }
  }
  // Second, compute center of mass
  center_of_mass = rigid.initialize_mass_and_inertia(density);
  if (!config.get("recenter", true)) {
    TC_ASSERT(rigid.pos_func);
    TC_ASSERT(rigid.rot_func);
    center_of_mass = Vector(0);
  }
  // if there is scripted pos
  if (rigid.pos_func) {
    // rigid.set_infinity_mass();  // removed
  }
  // if there is scripted rot
  if (rigid.rot_func) {
    rigid.set_infinity_inertia();
  }
  // Third, translate to make sure the mesh has its center of mass at the origin
  for (auto &elem : elements) {
    for (int k = 0; k < dim; k++) {
      elem.v[k] = elem.v[k] - center_of_mass;
    }
  }
  // Forth set rigid centoid to be origin, so that particles get correct offsets
  rigid.position = Vector(0.0_f);
  // 2d
  TC_STATIC_IF(dim == 2) {
    for (auto &elem : elements) {
      Vector a = elem.v[0], b = elem.v[1];
      int n_samples =
          std::max((int)std::ceil((a - b).length() * inv_delta_x), 2);
      for (int j = 0; j < n_samples; j++) {
        Vector pos = lerp((0.5_f + j) / n_samples, a, b);
        // NOTE: no "corner" boundary particle for 2D
        add_boundry_particle(pos, elem.get_normal(), &elem);
      }
    }
  }
  // 3d
  TC_STATIC_ELSE {
    rigid.rotation_axis = config.get("rotation_axis", Vector(0.0_f));
    for (auto &elem : elements) {
      std::vector<Vector> positions;
      Vector x_n = normalize(elem.v[1] - elem.v[0]);
      Vector y_n = normalize(elem.v[2] - elem.v[0]);
      real x_length = length(elem.v[1] - elem.v[0]);
      real y_length = length(elem.v[2] - elem.v[0]);
      for (real _x = min(x_length / 3.0_f, this->delta_x / 2.0_f);
           _x < x_length + this->delta_x; _x += this->delta_x)
        for (real _y = min(y_length / 3.0_f, this->delta_x / 2.0_f);
             _y < y_length + this->delta_x; _y += this->delta_x) {
          real x = ((_x < x_length) ? _x : _x - this->delta_x / 2.0_f);
          real y = ((_y < y_length) ? _y : _y - this->delta_x / 2.0_f);
          if (x / x_length + y / y_length > 1.0_f - eps)
            continue;
          Vector position = elem.v[0] + x_n * x + y_n * y;
          positions.push_back(position);
        }
      for (auto &position : positions)
        // add boundary particle
        add_boundry_particle(position, elem.get_normal(), &elem);
    }
    TC_TRACE("Mesh #elements = {}", mesh->elements.size());
  }
  TC_STATIC_END_IF

  // check rigid particle position ---------------------------------------------
  rigid.position = initial_position;
  for (auto &p_i : added_particles) {
    auto p = static_cast<RigidBoundaryParticle<dim> *>(allocator[p_i]);
    p->align_with_rigid_body();
    if (near_boundary(*p)) {
      TC_WARN(
          "Particle close to domain boundary detected when adding rigid body");
      continue;
    }
    particles.push_back(p_i);
  }

  rigids.push_back(std::move(rigid_ptr));
  TC_TRACE("#Particles: {}", particles.size());
}

// advect rigid bodies ---------------------------------------------------------
template <int dim>
void MPM<dim>::advect_rigid_bodies(real dt) {

  for (auto &r : rigids) {
    auto &rigid = *r;
    // rotation axis
    if (rigid.rotation_axis.abs_max() > 0.1_f) {
      rigid.enforce_angular_velocity_parallel_to(rigid.rotation_axis);
    }

    // rigid body gravity 
    // Frozen bodies are pinned (see freeze-on-axis-exit block below); applying
    // gravity to them would just create wasted impulses immediately undone by
    // the pin. Skip it.
    if (config_backup.get("rigidBody_gravity", true) && !rigid.is_frozen) {
      rigid.apply_impulse(this->gravity * rigid.get_mass() * dt, rigid.position);
    }

    // ADDED: externally applied (driving) force and torque on COM.
    // Only effective for dynamic bodies. For kinematic bodies (pos_func /
    // rot_func set) the imposed trajectory takes over in advance(), so we
    // skip the contribution.
    //
    // Each load can be specified as either a static Vector (applied_force /
    // applied_torque) or a function of time (applied_force_func /
    // applied_torque_func). If the function form is set, it is evaluated
    // at this->current_t to obtain the instantaneous load; otherwise the
    // static value is used. This lets the user gate, ramp, or otherwise
    // schedule arbitrary time-varying loads from Python without changing
    // the engine.
    //
    // Force on the COM: apply_impulse(impulse, position) uses the body's
    // position as the application point, so the arm (orig - position) is
    // zero and only velocity changes.
    //
    // Torque on the COM: apply_torque accepts AngularVelocity<dim>::ValueType
    // (scalar in 2D, Vector3 in 3D). The Vector form is adapted by
    // AppliedTorqueAdapter<dim>.
    if (!rigid.pos_func && !rigid.is_frozen) {
      // Evaluate force: function form takes precedence over static.
      Vector force_now = rigid.applied_force_func
                             ? rigid.applied_force_func(this->current_t)
                             : rigid.applied_force;
      if (force_now.abs_max() > 0.0_f) {
        rigid.apply_impulse(force_now * dt, rigid.position);
      }
    }
    if (!rigid.rot_func && !rigid.is_frozen) {
      // Evaluate torque: function form takes precedence over static.
      Vector torque_now = rigid.applied_torque_func
                              ? rigid.applied_torque_func(this->current_t)
                              : rigid.applied_torque;
      if (torque_now.abs_max() > 0.0_f) {
        rigid.apply_torque(
            AppliedTorqueAdapter<dim>::to_engine_torque(torque_now) * dt);
      }
    }

    int free_axis_in_position = config_backup.get("free_axis_in_position", 0); // added

    // advance
    rigid.advance(this->current_t, dt, free_axis_in_position);  // added

    // rotation axis
    if (rigid.rotation_axis.abs_max() > 0.1_f) {
      rigid.enforce_angular_velocity_parallel_to(rigid.rotation_axis);
    }

    // ADDED: lock specific linear / angular DOFs (only for dynamic bodies).
    // Pass e.g. lock_linear_axes=(0,0,1) and lock_angular_axes=(1,1,0) in the
    // MPM Python config to keep the body on a 2D plane (wheel rig).
    // A component > 0.5 means "lock this axis". For kinematic bodies the
    // trajectory is the source of truth, so we skip this.
    if (!rigid.pos_func) {
      Vector lock_lin =
          config_backup.get("lock_linear_axes", Vector(0.0_f));
      if (lock_lin.abs_max() > 0.5_f) {
        for (int i = 0; i < dim; i++) {
          if (lock_lin[i] > 0.5_f) {
            rigid.velocity[i] = 0.0_f;
          }
        }
      }
    }
    if (!rigid.rot_func) {
      Vector lock_ang =
          config_backup.get("lock_angular_axes", Vector(0.0_f));
      if (lock_ang.abs_max() > 0.5_f) {
        TC_STATIC_IF(dim == 3) {
          for (int i = 0; i < dim; i++) {
            if (lock_ang[i] > 0.5_f) {
              id(rigid.angular_velocity).value[i] = 0.0_f;
            }
          }
        }
        TC_STATIC_END_IF
      }
    }

    // ADDED: freeze-on-axis-exit safety net.
    // Opt-in per body via freeze_on_axis_exit=True. If enabled, check
    // whether the body's position along the chosen axis has left the
    // [min, max] interval. The first time it does, we LATCH is_frozen=true
    // (one-way) and capture the current pose (position + rotation) into
    // frozen_position / frozen_rotation. From that step on, the body is
    // effectively kinematic with a constant scripted pose:
    //   - position    <- frozen_position   (overrides any drift)
    //   - rotation    <- frozen_rotation   (overrides any drift)
    //   - velocity    <- 0
    //   - angular_vel <- 0
    // applied_* loads are already gated by is_frozen above, so the body
    // genuinely stops being driven. Gravity and soil-contact impulses still
    // computed by the engine but their effect is immediately cancelled by
    // the pin. Skipped entirely for already-kinematic bodies.
    if (rigid.freeze_on_axis_exit && !rigid.pos_func && !rigid.rot_func) {
      if (!rigid.is_frozen) {
        int axis = rigid.freeze_axis;
        if (axis >= 0 && axis < dim) {
          real p = rigid.position[axis];
          if (p < rigid.freeze_axis_min || p > rigid.freeze_axis_max) {
            rigid.is_frozen = true;
            // Capture the pose at the moment of triggering. We use position
            // and rotation as observed RIGHT NOW (after this step's advance,
            // any locks, etc.) so the pin matches what the user/network last
            // saw rendered.
            rigid.frozen_position = rigid.position;
            rigid.frozen_rotation = rigid.rotation;
            TC_INFO("Rigid body {} frozen at t = {}: position[{}] = {} "
                    "outside [{}, {}]; pinning pose.",
                    rigid.id, this->current_t, axis, p,
                    rigid.freeze_axis_min, rigid.freeze_axis_max);
          }
        }
      }
      if (rigid.is_frozen) {
        // Pin pose to the captured value. This overrides anything that
        // happened during this step (advance + contact + locks) and keeps
        // the body geometrically still, like a kinematic body with constant
        // scripted_position / scripted_rotation.
        rigid.position = rigid.frozen_position;
        rigid.rotation = rigid.frozen_rotation;
        rigid.velocity = Vector(0.0_f);
        TC_STATIC_IF(dim == 3) {
          id(rigid.angular_velocity).value = id(Vector(0.0_f));
        }
        TC_STATIC_ELSE {
          id(rigid.angular_velocity.value) = 0;
        }
        TC_STATIC_END_IF
      }
    }

    // print rigid body state
    if (config_backup.get("print_rigid_body_state", true)) {
      // TC_P(rigid->get_mass());
      // TC_P(rigid->get_inertia());
      TC_P(rigid.position);
      // TC_P(rigid.rotation.value);
      TC_P(rigid.velocity);
      TC_P(rigid.angular_velocity.value);
    }
  }
  // align particle with rigid body --------------------------------------------
  parallel_for_each_particle([](Particle &p_) {
    if (p_.is_rigid()) {
      auto *p = dynamic_cast<RigidBoundaryParticle<dim> *>(&p_);
      p->align_with_rigid_body();
    }
  });

}

// rigid body collision -------------------------------------------------- : OFF
template <int dim>
void MPM<dim>::rigidify(real dt) {
  if (!config_backup.get<bool>("rigid_body_collision", true)) {
    return;
  }
  std::vector<Collision<dim>> collisions;
  {
    Profiler _("collision detection");
    TC_STATIC_IF(dim == 3) {
      RigidSolver<dim> rigid_solver;
      rigid_solver.detect_rigid_collision(rigids, collisions);
    }
    TC_STATIC_END_IF
  }
  if (!collisions.size()) {
    return;
  }
  {
    Profiler _("collision resolution");
    int iterations = config_backup.get("rigid_body_iterations", 5);
    for (int i = 0; i < iterations; i++) {
      auto rigid_penalty = config_backup.get("rigid_penalty", 1e3_f);
      if (config_backup.get("rigid_body_position_iterations", true)) {
        for (auto &col : collisions) {
          col.project_position(dt, rigid_penalty);
        }
      }
      for (auto &col : collisions) {
        col.project_velocity();
      }
    }
    for (int i = 0; i < iterations; i++) {
      for (auto &col : collisions) {
        col.project_velocity();
      }
    }
  }
}

// rigid body-levelset collision ----------------------------------------- : OFF
template <int dim>
void MPM<dim>::rigid_body_levelset_collision(real t, real delta_t) {
  // particle
  for (auto &p_i : particles) {
    auto &p = *allocator[p_i];
    // rigid particle
    if (p.is_rigid()) {
      Vector pos        = p.pos * inv_delta_x;           // magnified pos
      real phi          = this->levelset.sample(pos, t); // levelset func
      Vector gradient   = this->levelset.get_spatial_gradient(pos, t);
      RigidBody<dim> *r = dynamic_cast<RigidBoundaryParticle<dim> *>(&p)->rigid;

      // if rigid particle is not in the desired place
      if (phi < 0) {
        real friction     = r->frictions[0];
        real cRestitution = r->restitution;
        Vector v10        = r->get_velocity_at(p.pos);
        Vector r0         = p.pos - r->position;
        real v0           = dot(gradient, v10);

        real J = -((1 + cRestitution) * v0) *
                 inversed(r->get_impulse_contribution(r0, gradient));
        if (J < 0) {
          continue;
        }
        Vector impulse = J * gradient;
        r->apply_impulse(impulse, p.pos);

        // Friction
        v10 = r->get_velocity_at(p.pos);
        Vector tao = v10 - gradient * dot(gradient, v10);
        if (tao.abs_max() > 1e-7_f) {
          tao = normalized(tao);
          real j = -dot(v10, tao) / (r->get_impulse_contribution(r0, tao));
          j = clamp(j, friction * -J, friction * J);
          Vector fImpulse = j * tao;
          r->apply_impulse(fImpulse, p.pos);
        }
      }
    }
  }
}

template void MPM<2>::rigidify(real dt);
template void MPM<3>::rigidify(real dt);
template std::unique_ptr<RigidBody<2>> MPM<2>::create_rigid_body(Config config);
template std::unique_ptr<RigidBody<3>> MPM<3>::create_rigid_body(Config config);
template void MPM<2>::advect_rigid_bodies(real dt);
template void MPM<3>::advect_rigid_bodies(real dt);
template void MPM<2>::rigid_body_levelset_collision(real t, real delta_t);
template void MPM<3>::rigid_body_levelset_collision(real t, real delta_t);
template void MPM<2>::add_rigid_particle(Config config);
template void MPM<3>::add_rigid_particle(Config config);
TC_NAMESPACE_END
