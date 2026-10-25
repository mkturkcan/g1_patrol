#!/bin/bash
# Software-in-the-loop test: the whole navigation stack (odometry bridge, SLAM, planner, policy, shield, Unitree
# sport requests) against the simulated B2-W in a lab world, on this computer. ROS 2 (Foxy or Humble) and
# unitree_ros2's messages must be sourced; run from the repository root:
#
#   bash deploy/b2w/sim/run_sil.sh [goal_x goal_y] [start_x start_y start_yaw] [people_json]
#
# Prints "SIL RESULT {...}" and exits with 0 if the robot reached the goal and never moved into anything. FAKE_ARGS adds
# parameters of the simulated robot (e.g. "-p deadman_held:=false"), SIL_TIMEOUT sets the time allowed (s).
set -u
GX=${1:-4.0}; GY=${2:--3.5}
SX=${3:--6.5}; SY=${4:-0.0}; SYAW=${5:-0.0}
PEOPLE=${6:-}
ROOT=$(pwd)
export PYTHONPATH="$ROOT/src:$ROOT/deploy:${PYTHONPATH:-}"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-73}
LOG=${SIL_LOG:-/tmp/b2w_sil}
mkdir -p "$LOG"
PEOPLE_ARG=()
[ -n "$PEOPLE" ] && PEOPLE_ARG=(-p "people:='$PEOPLE'")  # a YAML string
setsid python3 -m b2w.sim.fake_b2w --ros-args -p world:="$ROOT/deploy/b2w/sim/lab_world.json" \
    -p start:="[$SX, $SY, $SYAW]" "${PEOPLE_ARG[@]}" ${FAKE_ARGS:-} > "$LOG/fake_b2w.log" 2>&1 &
FAKE=$!
setsid ros2 launch "$ROOT/deploy/b2w/launch/b2w_navigation.launch.py" mode:=drive lidar_driver:=false cameras:=false \
    max_speed:=0.6 > "$LOG/stack.log" 2>&1 &
STACK=$!
python3 -m b2w.sim.sil_check --ros-args -p goal:="[$GX, $GY]" -p trace:="$LOG/trace.json" -p timeout:=${SIL_TIMEOUT:-150.0} 2>&1 | tee "$LOG/check.log"
CODE=${PIPESTATUS[0]}
# each started in its own process group: stop the groups (the launch and the nodes it started)
kill -INT -- -$STACK -$FAKE 2>/dev/null
sleep 5
kill -9 -- -$STACK -$FAKE 2>/dev/null
echo "logs: $LOG (fake_b2w.log, stack.log, check.log)"
exit $CODE
