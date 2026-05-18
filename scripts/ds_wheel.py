## Wheel-soil model prepared for generating training dataset
## Amin Haeri [ahaeri92@gmail.com]
## Mid 2021
##
## Modified: supports two simulation modes with unified frame structure for
## downstream tfrecord generation (run_tfrecord_wheel_unified.py).
##
##   mode = 0  (kinematic) -> original behaviour: prescribed slip,
##                            scripted_position + scripted_rotation.
##   mode = 1  (dynamic)   -> torque-driven, two-way wheel-soil coupling.
##                            Trajectory and slip emerge from contact physics.
##
## FRAME STRUCTURE (the same in BOTH modes -- this is what the tfrecord
## generator depends on):
##   frames 0..240    : rest phase. The wheel sits above the soil, settles
##                      under gravity. No prescribed motion, no torque.
##                      In dynamic mode the torque function applied_torque_fn
##                      returns the zero vector while t <= time_rest_sim.
##   frames 241..end  : working phase.
##                      - kinematic: wheel translates in -x and rotates,
##                                   according to scripted_position / rotation.
##                      - dynamic:   driving torque activates; wheel rolls
##                                   under physics + soil reaction.
##   Both modes share frames 0..240 (physically equivalent: pure settling).
##   We aim for at least 561 total frames (rest + 320 working), matching the
##   tfrecord generator's `frames = min(320, len(files)-241)` assumption.
##
## SAFETY (dynamic mode only):
##   The wheel may otherwise reach the left wall of the bin if the applied
##   torque is large or the working phase is long. To prevent this, the
##   engine's freeze_on_axis_exit mechanism is enabled: once the wheel's x
##   coordinate drops below x_safety_edge, the engine "freezes" the wheel
##   (pose pinned, velocities zeroed, applied torque ignored) for the rest
##   of the simulation. The MPM grid + soil continue to be simulated and
##   tfrecord frames keep being produced.
##
## CSV OUTPUT FORMAT is the same as the original (write_dataset=True).
## TASK_ID format is unified: <c>_M<mode>_G<gravity>ms-2_S<slip>perc_T<torque>Nm
##                            _L<load>N_D<dia>cm_A<sfangle>deg
## Both slip and torque appear in the name; the inactive one is 0.
##
## Requires the engine extension that adds:
##   - applied_force / applied_torque (static Vector loads)
##   - applied_force_fn / applied_torque_fn (time-varying loads via the same
##     Function13 mechanism as scripted_position / scripted_rotation)
##   - lock_linear_axes / lock_angular_axes (per-axis DOF locks)
##   - freeze_on_axis_exit (latch the rigid body in place when its position
##     along one axis leaves a configured interval)
## See patched rigid_body.h, mpm_rigid_body.cpp, and mpm.py.

# MPM          EXP
# 1 u      ->  4 m
# 0.03125  <-  0.125 m (wheel's width)

#!/usr/bin/env python

import sys
import math

import taichi


# =====================================================================
# Frame-budget tunables
# =====================================================================
# These mirror the assumptions hard-coded in run_tfrecord_wheel_pca.py:
#   - REST_FRAMES kicks in for `range(241, ...)` in the tfrecord reader,
#     so the rest phase is 0..REST_FRAMES inclusive (REST_FRAMES+1 frames).
#   - WORKING_FRAMES is the count the tfrecord generator wants to read.
# The simulation length is set so the engine produces strictly more than
# REST_FRAMES + WORKING_FRAMES output frames, with a small extra margin.
REST_FRAMES     = 240
WORKING_FRAMES  = 320
EXTRA_MARGIN    = 20   # a few spare frames so off-by-one never bites

# Auto-stop edge margin (multiple of wheel radius) -- only used to *cap* the
# working phase length if a fast wheel would otherwise hit the bin wall.
# It is checked AGAINST the working-phase budget; whichever is smaller wins.
EDGE_MARGIN_FACTOR = 0.8

# Hard ceiling on real-world simulated time (seconds). Last-resort cap.
TIME_TOTAL_MAX_REAL = 30.0


if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Dimensional analysis (unchanged)
    # ------------------------------------------------------------------
    # Convention: every "scale_*" is the multiplicative factor that takes a
    # real-world quantity to the corresponding simulation-units quantity:
    #     X_sim = X_real * scale_X
    # The chosen scales below imply consistent units for derived quantities:
    #     L_sim = L_real * scale_dim                  (scale_dim   = 0.25)
    #     T_sim = T_real * scale_time                 (scale_time  = 0.5)
    #     V_sim = V_real * scale_lin_vel              (= scale_dim/scale_time = 0.5)
    #     W_sim = W_real * scale_ang_vel              (= 1/scale_time = 2)
    #     M_sim = M_real * scale_dim^3                (mass: density in real kg/m^3
    #                                                  is fed to the engine; the
    #                                                  engine integrates over a
    #                                                  sim-units volume, so the
    #                                                  effective mass scales by L^3)
    #     a_sim = a_real * 1                          (acceleration is invariant
    #                                                  because scale_dim = scale_time^2
    #                                                  for this particular choice;
    #                                                  hence gravity is passed
    #                                                  numerically unchanged)
    #     F_sim = F_real * scale_force                (= scale_dim^3 = 0.015625)
    #     tau_sim = tau_real * scale_torque           (= scale_dim^4 = 0.00390625)
    scale_dim     = 0.25
    scale_time    = scale_dim * 2
    scale_lin_vel = scale_dim * 2
    scale_ang_vel = scale_lin_vel / scale_dim
    scale_force   = scale_dim**3
    scale_torque  = scale_dim**4

    # ------------------------------------------------------------------
    # Command-line inputs.
    #
    # Usage:
    #   python ds_wheel.py <c> <gravity> <mode> <param3> <wload> <wdia_cm> <sfangle>
    #
    #   mode    = 0  (kinematic) -> param3 is slip [%]: 20, 40, or 70
    #   mode    = 1  (dynamic)   -> param3 is applied torque [N*m]
    # ------------------------------------------------------------------
    c       = int(sys.argv[1])           # run id / counter
    gravity = float(sys.argv[2])         # [m/s^2]
    mode    = int(sys.argv[3])           # 0 = kinematic, 1 = dynamic
    param3  = float(sys.argv[4])         # slip (%) or torque (N*m)
    wload   = float(sys.argv[5])         # [N]
    wdia    = float(sys.argv[6]) / 100   # [m]
    sfangle = float(sys.argv[7])         # [deg]

    assert mode in (0, 1), "mode must be 0 (kinematic) or 1 (dynamic)"
    mode_name = 'kinematic' if mode == 0 else 'dynamic'

    print('inputs  =', c, gravity, mode, '(' + mode_name + ')',
          param3, wload, wdia, sfangle)

    # ------------------------------------------------------------------
    # Numerical discretization
    # ------------------------------------------------------------------
    r   = 301
    dx  = 1.0 / r
    dt  = 1e-5
    ppc = 4
    frame_rate = 60 / scale_time     # [Hz]
    frame_dt   = 1.0 / frame_rate

    # ------------------------------------------------------------------
    # Bin (sim units)
    # ------------------------------------------------------------------
    friction_ls = -1
    offset      = 0.05 / 2
    x_bin = 0.400     * scale_dim
    y_bin = 0.350 / 6 * scale_dim
    z_bin = 0.190     * scale_dim

    # ------------------------------------------------------------------
    # Soil
    # ------------------------------------------------------------------
    modulus_elastic   = 15e6 * scale_dim
    ratio_poisson     = 0.3
    modulus_shear     = modulus_elastic / 2 / (1 + ratio_poisson)
    modulus_bulk      = modulus_elastic / 3 / (1 - 2 * ratio_poisson)
    friction_static   = math.tan(math.pi / 180 * sfangle)
    friction_2        = 0.9616
    grain_diameter    = 0.0003 * scale_dim
    grain_density     = 2583
    packing_fraction  = 0.67
    critical_density  = packing_fraction * grain_density

    # ------------------------------------------------------------------
    # Wheel geometry & mass
    # ------------------------------------------------------------------
    wwidth = 0.125

    # Base nominal angular velocity (used in KINEMATIC mode only, kept as in
    # the original script).
    angular_velocity = 0.13   # [rad/s]

    if wdia == 0.15:
        wwidth = wwidth * 2/3
        angular_velocity = angular_velocity * 2
    elif wdia == 0.05:
        wwidth = wwidth * 2/3
        wload  = wload  / 9
        angular_velocity = angular_velocity * 6
        if mode == 1:
            # Dynamic mode: param3 is torque -- scale it down with load.
            param3 = param3 / 9

    wheel_volume  = math.pi * (wdia / 2)**2 * wwidth
    wheel_density = wload / wheel_volume / gravity

    friction_rb = -5

    # ------------------------------------------------------------------
    # Mode-specific control values
    # ------------------------------------------------------------------
    # We always derive BOTH slip and torque variables (one is zero for the
    # inactive mode). This lets the task_id, the diagnostics, and the eventual
    # step_context in the tfrecord be uniform across modes.
    slip_value         = 0.0   # %
    horizontal_v_real  = 0.0   # m/s -- for kinematic scripted velocity
    horizontal_v_sim   = 0.0
    torque_value       = 0.0   # N*m
    applied_torque_sim = 0.0

    if mode == 0:
        # Kinematic mode: param3 is slip in percent.
        slip_value = param3
        if   slip_value == 20: horizontal_v_real = 0.01560
        elif slip_value == 40: horizontal_v_real = 0.01200
        elif slip_value == 70: horizontal_v_real = 0.00585
        else:
            raise ValueError("slip must be 20, 40, or 70 (percent)")
        horizontal_v_sim = horizontal_v_real * scale_lin_vel
        # Scripted rotation rate in deg/s (engine expects degrees there).
        angular_velocity_deg_sim = (
            180 / math.pi * angular_velocity * scale_ang_vel
        )
    else:
        # Dynamic mode: param3 is applied torque in N*m (real).
        torque_value = param3
        applied_torque_sim = torque_value * scale_torque

    # ------------------------------------------------------------------
    # Wheel placement (sim units)
    # ------------------------------------------------------------------
    wheel_diameter = wdia * scale_dim
    wheel_radius   = wheel_diameter / 2
    x_wheel = offset + x_bin * 0.7
    y_wheel = offset + y_bin + wheel_radius + dx
    z_wheel = offset + z_bin * 0.5
    scale_x_wheel = scale_dim * (wdia   / 0.30)
    scale_y_wheel = scale_dim * (wdia   / 0.30)
    scale_z_wheel = scale_dim * (wwidth / 0.125)

    # ------------------------------------------------------------------
    # Phase lengths in sim seconds
    # ------------------------------------------------------------------
    # The rest phase MUST give the tfrecord generator >= 240 frames.
    # We compute durations from the desired frame count and frame_rate.
    time_rest_sim     = REST_FRAMES    / frame_rate         # ~ 240/120 sim s
    time_working_sim  = WORKING_FRAMES / frame_rate         # ~ 320/120 sim s
    time_margin_sim   = EXTRA_MARGIN   / frame_rate

    # ------------------------------------------------------------------
    # Auto-stop check: would the wheel hit the bin wall before
    # time_working_sim is over?
    # ------------------------------------------------------------------
    # Wheel moves in -x. Stop when |x_wheel - left_wall| would shrink below
    # EDGE_MARGIN_FACTOR * R. Use a representative horizontal velocity:
    #   * kinematic: exactly horizontal_v_sim
    #   * dynamic  : upper bound via free-spin assumption (omega_ref * R).
    x_safety_edge = offset + EDGE_MARGIN_FACTOR * wheel_radius
    travel_budget = x_wheel - x_safety_edge
    assert travel_budget > 0, "wheel starts past safety edge -- adjust x_wheel"

    if mode == 0:
        v_char_sim = horizontal_v_sim
    else:
        omega_ref_real = max(angular_velocity, 0.13)
        v_char_real    = omega_ref_real * (wdia / 2.0)
        v_char_sim     = v_char_real * scale_lin_vel

    if v_char_sim > 0:
        time_to_edge_sim_working = travel_budget / v_char_sim
    else:
        time_to_edge_sim_working = float('inf')

    # Clip the working phase by time_to_edge only in KINEMATIC mode. In
    # DYNAMIC mode the engine's freeze_on_axis_exit safety net (configured
    # below for the rigid body) takes care of the wheel reaching the wall:
    # the wheel is simply stopped in place and the simulation continues to
    # generate frames normally. So in dynamic we always go for the full
    # working phase length.
    if mode == 0:
        time_working_actual_sim = min(time_working_sim,
                                      time_to_edge_sim_working)
        if time_working_actual_sim < time_working_sim:
            print('!! WARNING: kinematic working phase clipped from {:.3f} to '
                  '{:.3f} sim s (wheel would reach edge)'.format(
                      time_working_sim, time_working_actual_sim))
    else:
        # Dynamic: trust the engine-level freeze to stop the wheel. Always
        # produce the full working-phase length so the tfrecord generator
        # gets its 320 frames regardless of whether/when the wheel freezes.
        time_working_actual_sim = time_working_sim

    # Apply the hard ceiling (real-world seconds).
    time_total_max_sim = TIME_TOTAL_MAX_REAL * scale_time

    # Final total simulated time and frame count.
    time_total_sim = min(
        time_rest_sim + time_working_actual_sim + time_margin_sim,
        time_total_max_sim,
    )
    num_frames = int(math.ceil(time_total_sim * frame_rate))
    num_frames = max(num_frames, REST_FRAMES + 1)  # at least one working frame

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    print('--- run setup ---')
    print('  mode                  : {} ({})'.format(mode, mode_name))
    print('  wheel radius (sim)    :', wheel_radius)
    print('  x_wheel start         :', x_wheel)
    print('  x_safety_edge         :', x_safety_edge)
    print('  travel_budget (sim)   :', travel_budget)
    print('  v_char (sim)          :', v_char_sim)
    print('  time_rest (sim)       :', time_rest_sim)
    print('  time_working (sim)    :', time_working_actual_sim)
    print('  time_total (sim)      :', time_total_sim)
    print('  num_frames            :', num_frames)
    if mode == 0:
        print('  slip (%)              :', slip_value)
        print('  horizontal v (real)   :', horizontal_v_real, 'm/s')
        print('  angular v (rad/s)     :', angular_velocity)
    else:
        tau_threshold_real = (wdia / 2.0) * friction_static * wload
        # Estimate the wheel's free-spin angular acceleration under the applied
        # torque (i.e. in vacuum, no soil reaction). This is an UPPER BOUND on
        # the actual angular acceleration during simulation; soil contact will
        # reduce it. Useful as a sanity check that the torque produces a
        # physically reasonable rotation rate within the working-phase time.
        wheel_mass_real      = wload / gravity                       # [kg]
        wheel_inertia_real   = 0.5 * wheel_mass_real * (wdia/2)**2   # solid disk I = 1/2 m R^2
        alpha_free_real      = torque_value / wheel_inertia_real     # [rad/s^2]
        time_to_steady_real  = angular_velocity / alpha_free_real if alpha_free_real > 0 else float('inf')
        print('  applied torque (real) :', torque_value, 'N*m')
        print('  applied torque (sim)  :', applied_torque_sim, 'sim N*m')
        print('  start threshold (real):', tau_threshold_real, 'N*m')
        if tau_threshold_real > 0:
            print('  applied / threshold   :', torque_value / tau_threshold_real)
        print('  wheel mass (real)     :', wheel_mass_real,    'kg')
        print('  wheel inertia (real)  :', wheel_inertia_real, 'kg*m^2  (solid disk)')
        print('  free-spin alpha (real):', alpha_free_real,    'rad/s^2  (UPPER bound)')
        print('  time to ref omega     :', time_to_steady_real, 's       (UPPER bound)')
    print('-----------------')

    # ------------------------------------------------------------------
    # MPM simulator
    # ------------------------------------------------------------------
    # Unified task_id: both slip and torque appear, inactive one is 0.
    # M<mode> prefix makes the mode immediately visible in directory names.
    task_tag = (
        str(c) +
        '_M' + str(mode) +
        '_G' + str(gravity) + 'ms-2' +
        '_S' + str(int(slip_value)) + 'perc' +
        '_T' + str(round(torque_value, 3)) + 'Nm' +
        '_L' + str(int(wload)) + 'N' +
        '_D' + str(int(wdia*100)) + 'cm' +
        '_A' + str(int(sfangle)) + 'deg'
    )

    mpm_kwargs = dict(
        task_id=task_tag,
        res=(r, r, r),
        base_delta_t=dt,
        frame_dt=frame_dt,
        num_frames=num_frames,
        num_threads=-1,
        gravity=(0, -gravity, 0),
        particle_gravity=True,
        rigidBody_gravity=True,
        rigid_body_collision=False,
        rigid_body_levelset_collision=False,
        particle_collision=False,
        pushing_force=0,
        print_rigid_body_state=False,
        clean_boundary=False,
        warn_particle_deletion=False,
        cdf_3d_modified=True,
        compute_particle_impulses=True,
        visualize_particle_impulses=False,
        affect_particle_impulses=True,
        particle_bc_at_levelset=False,
        snapshots=False,
        verbose_bgeo=False,
        write_particle=False,
        write_rigid_body=False,
        write_partio=False,
        write_dataset=True,
    )

    if mode == 1:
        # Dynamic: engine-level DOF locks. The wheel can only translate in
        # x (forward/backward) and y (up/down), and rotate about z (spin
        # axis). All other DOFs locked at integrator level to prevent
        # lateral drift, tip-over, and yawing.
        mpm_kwargs.update(dict(
            lock_linear_axes=(0, 0, 1),
            lock_angular_axes=(1, 1, 0),
            free_axis_in_position=0,
        ))
    else:
        # Kinematic: original engine behaviour. y is the free axis, wheel
        # follows scripted_position / scripted_rotation in x and z.
        mpm_kwargs.update(dict(
            free_axis_in_position=2,
        ))

    mpm = taichi.dynamics.MPM(**mpm_kwargs)

    # ------------------------------------------------------------------
    # Level set: bin walls
    # ------------------------------------------------------------------
    levelset = mpm.create_levelset()
    levelset.add_plane(taichi.Vector( 1, 0, 0), -offset)
    levelset.add_plane(taichi.Vector( 0, 1, 0), -offset)
    levelset.add_plane(taichi.Vector( 0, 0, 1), -offset)
    levelset.add_plane(taichi.Vector(-1, 0, 0),  offset + x_bin)
    levelset.add_plane(taichi.Vector( 0, 0,-1),  offset + z_bin)
    levelset.set_friction(friction_ls)
    mpm.set_levelset(levelset, False)

    # ------------------------------------------------------------------
    # Soil particles
    # ------------------------------------------------------------------
    tex = taichi.Texture(
        'mesh',
        scale=taichi.Vector(x_bin*10, y_bin*10, z_bin*10),
        translate=(offset + x_bin/2, offset + y_bin/2, offset + z_bin/2),
        resolution=(2*r, 2*r, 2*r),
        mesh_accuracy=3,
        filename='projects/mpm/data/cube_smooth.obj',
    ) * ppc

    mpm.add_particles(
        type='nonlocal',
        pd=True,
        density_tex=tex.id,
        density=grain_density,
        critical_density=critical_density,
        packing_fraction=packing_fraction,
        S_mod=modulus_shear,
        B_mod=modulus_bulk,
        dia=grain_diameter,
        mu_s=friction_static,
        mu_2=friction_2,
        A_mat=0.48,
        I_0=0.278,
        t_0=1e-4,
    )

    # ------------------------------------------------------------------
    # Wheel particle
    # ------------------------------------------------------------------
    if mode == 0:
        # KINEMATIC -- original Haeri behaviour.
        # Rest then translate. Position is fully prescribed in x and z
        # (with y free to settle); rotation is fully prescribed.
        def position_function(t):
            if t <= time_rest_sim:
                return taichi.Vector(x_wheel, y_wheel, z_wheel)
            else:
                return taichi.Vector(
                    x_wheel - horizontal_v_sim * (t - time_rest_sim),
                    y_wheel,
                    z_wheel,
                )

        def rotation_function(t):
            if t <= time_rest_sim:
                return taichi.Vector(0, 0, 0)
            else:
                return taichi.Vector(
                    0, 0, angular_velocity_deg_sim * (t - time_rest_sim),
                )

        mpm.add_particles(
            type='rigid',
            linear_damping=1e3,
            density=wheel_density,
            packing_fraction=1,
            friction=friction_rb,
            scripted_position=taichi.function13(position_function),
            scripted_rotation=taichi.function13(rotation_function),
            scale=(scale_x_wheel, scale_y_wheel, scale_z_wheel),
            codimensional=False,
            mesh_fn='projects/mpm/data/wheel_houdini.obj',
        )

    else:
        # DYNAMIC -- function-gated torque.
        # The driving torque is delivered via `applied_torque_fn`, a function
        # of simulation time (same Function13 signature as scripted_position).
        # During the rest phase it returns the zero vector; after time_rest_sim
        # it returns the constant drive vector. This matches the kinematic
        # mode's "settle then move" structure exactly:
        #
        #   * frames 0..240   identical in both modes -- the wheel sits on
        #     the soil under gravity, no externally imposed load;
        #   * frames 241+     diverge -- kinematic follows scripted_position,
        #     dynamic experiences the activated torque, slip is emergent.
        #
        # Using a function rather than a static value lets the engine see a
        # genuine zero load during rest, exactly as kinematic mode does, and
        # would also support more elaborate schedules later (e.g. ramping
        # torque, sinusoidal drive) without further engine changes.

        def torque_function(t):
            if t <= time_rest_sim:
                return taichi.Vector(0, 0, 0)
            else:
               return taichi.Vector(0, 0, applied_torque_sim)

        # applied_force_fn left unset -- no external linear drive in this
        # experiment (the wheel's only "force" input is its own weight via
        # rigidBody_gravity=True).

        mpm.add_particles(
            type='rigid',
            linear_damping=0,
            angular_damping=0,
            density=wheel_density,
            packing_fraction=1,
            friction=friction_rb,
            # Time-varying torque (function form). The static applied_torque
            # is left at zero so it has no effect; the function dominates.
            applied_torque_fn=taichi.function13(torque_function),
            # Freeze-on-axis-exit safety net. If the wheel's x position drops
            # below x_safety_edge (close to the bin's left wall), the engine
            # latches the body as "frozen": from that step on, pose is pinned
            # to the captured value (position + rotation), linear and angular
            # velocities are zeroed, and applied_torque_fn is no longer
            # evaluated. Gravity on the body is also skipped (it would be
            # immediately undone by the pin). Soil contact impulses are still
            # accumulated by the transfer step but are overwritten at the end
            # of advect_rigid_bodies, so the body really stays put. The MPM
            # simulation as a whole continues -- the soil keeps reacting to
            # the wheel as a static obstacle, and tfrecord frames keep being
            # produced. We give a generous upper bound so only the LEFT edge
            # triggers the freeze.
            freeze_on_axis_exit=True,
            freeze_axis=0,                       # 0 = x
            freeze_axis_min=x_safety_edge,
            freeze_axis_max=1.0e30,              # effectively +inf
            initial_position=(x_wheel, y_wheel, z_wheel),
            initial_rotation=(0, 0, 0),
            initial_velocity=(0, 0, 0),
            initial_angular_velocity=(0, 0, 0),
            scale=(scale_x_wheel, scale_y_wheel, scale_z_wheel),
            codimensional=False,
            mesh_fn='projects/mpm/data/wheel_houdini.obj',
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    mpm.simulate(
        clear_output_directory=True,
        print_profile_info=False,
    )
