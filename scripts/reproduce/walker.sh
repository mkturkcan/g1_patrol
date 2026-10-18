#!/bin/bash
# Three-phase training of the locomotion policy (used by 1_walker.sh and 5_ablations.sh):
#   walker.sh <run tag> [extra Hydra overrides, applied in every phase]
#   A  10,000 iterations from scratch with moderate randomization: friction 0.6-1.2, no payload, rated motor torque,
#      actuation latency up to 10 ms, no sensor biases;
#   B   8,000 iterations from A with the task's full randomization on ground of friction 1.0: MuJoCo-Warp uses the
#      larger coefficient of a contact, so the contact friction is at least 1.0; shoves ramp up to full strength;
#   C   3,000 iterations from B on the task's ground (friction 0.1, combined by maximum), so that the contact friction is
#      the robot's randomized value (0.3-1.2); shoves at full strength.
# Newton (MuJoCo-Warp), 16,384 robots on 4 GPUs. The final policy is logs/rsl_rl/g1_walk_city/<run tag>/model_*.pt.
# A phase whose run directory holds a .complete marker is skipped, so an interrupted run can be continued.
source scripts/reproduce/env.sh
RUN_TAG=$1
shift
EXTRA=("$@")
MODERATE=(
    "env.events.physics_material.params.static_friction_range=[0.6,1.2]"
    "env.events.physics_material.params.dynamic_friction_range=[0.5,1.0]"
    "env.events.add_base_mass.params.mass_distribution_params=[0.0,0.0]"
    "env.events.motor_strength.params.strength_range=[1.0,1.0]"
    env.scene.robot.actuators.legs.max_delay=2 env.scene.robot.actuators.feet.max_delay=2
)
for term in base_ang_vel projected_gravity joint_pos; do
    MODERATE+=("env.observations.policy.$term.noise.bias_noise_cfg.n_min=0.0" "env.observations.policy.$term.noise.bias_noise_cfg.n_max=0.0")
done
HIGH_FRICTION=()
for material in env.scene.terrain.physics_material env.sim.physics_material; do
    HIGH_FRICTION+=("$material.static_friction=1.0" "$material.dynamic_friction=1.0" "$material.friction_combine_mode=multiply")
done
FULL_SHOVES=(env.curriculum.push_strength.params.modify_params.start_step=0
             env.curriculum.push_strength.params.modify_params.end_step=1)
COMMON=(isaaclab train_multigpu --num_gpus "$NUM_GPUS" --external_callback g1_patrol.register_tasks --seed "$SEED"
        --task G1Patrol-Walk-City --num_envs 4096)

phase() {  # phase <run directory name> <log name> <master port> <arguments...>
    local name=$1 log=$2 port=$3
    shift 3
    if [ -f "$LOGS/g1_walk_city/$name/.complete" ]; then
        echo "walker: $name already complete"
        return
    fi
    "${COMMON[@]}" --run_timestamp "$name" --master_port "$port" "$@" > "$OUT/$log.log" 2>&1
    touch "$LOGS/g1_walk_city/$name/.complete"
}

phase "${RUN_TAG}_phase_a" "walker_${RUN_TAG}_a" 29601 agent.max_iterations=10000 "${MODERATE[@]}" "${EXTRA[@]}"
phase "${RUN_TAG}_phase_b" "walker_${RUN_TAG}_b" 29602 --checkpoint "$(latest "$LOGS/g1_walk_city/${RUN_TAG}_phase_a")" \
    agent.max_iterations=8000 "${HIGH_FRICTION[@]}" "${EXTRA[@]}"
phase "$RUN_TAG" "walker_${RUN_TAG}_c" 29603 --checkpoint "$(latest "$LOGS/g1_walk_city/${RUN_TAG}_phase_b")" \
    agent.max_iterations=3000 "${FULL_SHOVES[@]}" "${EXTRA[@]}"
echo "walker $RUN_TAG: $(latest "$LOGS/g1_walk_city/$RUN_TAG")"
