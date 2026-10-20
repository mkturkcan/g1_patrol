#!/bin/bash
# Three-phase training of the locomotion policy (used by 1_walker.sh and 5_ablations.sh):
#   walker.sh <run tag> [extra Hydra overrides, applied in every phase]
#   A  10,000 iterations from scratch with moderate randomization: friction 0.6-1.2, no payload (torso or hands), rated
#      motor torque, actuation latency up to 10 ms, no sensor biases;
#   B   8,000 iterations from A with the task's full randomization on ground of friction 1.0: MuJoCo-Warp uses the
#      larger coefficient of a contact, so the contact friction is at least 1.0; shoves ramp up to full strength;
#   C   3,000 iterations from B on the task's ground (friction 0.1, combined by maximum), so that the contact friction is
#      the robot's randomized value (0.3-1.2); shoves at full strength.
# Newton (MuJoCo-Warp), 16,384 robots on 4 GPUs. WALKER_TASK selects the task (default G1Patrol-Walk-Reach, the walker that
# also squats and leans to pick things up; G1Patrol-Walk-City drives all joints and only walks). The final policy is
# logs/rsl_rl/<experiment>/<run tag>/model_*.pt (experiment g1_walk_reach or g1_walk_city).
# A phase whose run directory holds a .complete marker is skipped, so an interrupted run can be continued.
source scripts/reproduce/env.sh
RUN_TAG=$1
shift
WALKER_TASK=${WALKER_TASK:-G1Patrol-Walk-Reach}
case "$WALKER_TASK" in
    G1Patrol-Walk-Reach) EXP=g1_walk_reach ;;
    G1Patrol-Walk-City) EXP=g1_walk_city ;;
    *) echo "unknown walker task $WALKER_TASK"; exit 1 ;;
esac
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
if [ "$WALKER_TASK" = G1Patrol-Walk-Reach ]; then  # no loads in the hands either, and lighter shoves and kicks
    MODERATE+=("env.actions.disturbance.hand_load_mass=[0.0,0.0]" "env.actions.disturbance.shove_force=[30.0,120.0]"
               "env.actions.disturbance.kick_force=[30.0,100.0]")
fi
HIGH_FRICTION=()
for material in env.scene.terrain.physics_material env.sim.physics_material; do
    HIGH_FRICTION+=("$material.static_friction=1.0" "$material.dynamic_friction=1.0" "$material.friction_combine_mode=multiply")
done
FULL_SHOVES=(env.curriculum.push_strength.params.modify_params.start_step=0
             env.curriculum.push_strength.params.modify_params.end_step=1)
COMMON=(isaaclab train_multigpu --num_gpus "$NUM_GPUS" --external_callback g1_patrol.register_tasks --seed "$SEED"
        --task "$WALKER_TASK" --num_envs 4096)

phase() {  # phase <run directory name> <log name> <master port> <arguments...>
    local name=$1 log=$2 port=$3
    shift 3
    if [ -f "$LOGS/$EXP/$name/.complete" ]; then
        echo "walker: $name already complete"
        return
    fi
    "${COMMON[@]}" --run_timestamp "$name" --master_port "$port" "$@" > "$OUT/$log.log" 2>&1
    touch "$LOGS/$EXP/$name/.complete"
}

phase "${RUN_TAG}_phase_a" "${EXP}_${RUN_TAG}_a" 29601 agent.max_iterations=10000 "${MODERATE[@]}" "${EXTRA[@]}"
phase "${RUN_TAG}_phase_b" "${EXP}_${RUN_TAG}_b" 29602 --checkpoint "$(latest "$LOGS/$EXP/${RUN_TAG}_phase_a")" \
    agent.max_iterations=8000 "${HIGH_FRICTION[@]}" "${EXTRA[@]}"
phase "$RUN_TAG" "${EXP}_${RUN_TAG}_c" 29603 --checkpoint "$(latest "$LOGS/$EXP/${RUN_TAG}_phase_b")" \
    agent.max_iterations=3000 "${FULL_SHOVES[@]}" "${EXTRA[@]}"
echo "walker $RUN_TAG: $(latest "$LOGS/$EXP/$RUN_TAG")"
