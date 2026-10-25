#!/bin/bash
# Install the B2-W navigation stack on the robot's Jetson (Ubuntu 20.04 + ROS 2 Foxy, or 22.04 + Humble).
#
#   bash deploy/b2w/install.sh <network interface to the robot, e.g. eth0>
#
# Run from the repository root (git clone https://github.com/mkturkcan/g1_patrol.git). It installs (with sudo apt) the
# ROS 2 packages the stack uses -- slam_toolbox, pointcloud_to_laserscan, CycloneDDS, realsense2_camera -- and the
# Python packages (numpy, scipy, onnxruntime; PyTorch if missing), builds Unitree's message packages (unitree_go,
# unitree_api) and RoboSense's lidar driver (rslidar_sdk) in ~/b2w_ws, and writes ~/b2w_env.sh, which sets up a
# shell for the stack. Steps already done are skipped; run it again after a failure.
set -euo pipefail
IFACE=${1:-}
if [ -z "$IFACE" ]; then
    echo "usage: bash deploy/b2w/install.sh <network interface to the robot>"
    echo "interfaces with an address on the robot's network (192.168.123.x):"
    ip -4 -o addr | awk '{print "  " $2 " " $4}' | grep "192.168.123." || echo "  (none: connect to the robot first)"
    exit 1
fi
REPO=$(pwd)
[ -f "$REPO/deploy/b2w/launch/b2w_navigation.launch.py" ] || { echo "run from the g1_patrol repository root"; exit 1; }
WS=${WS:-$HOME/b2w_ws}
. /etc/os-release
case "$VERSION_ID" in
    20.04) DISTRO=foxy ;;
    22.04) DISTRO=humble ;;
    *) echo "Ubuntu $VERSION_ID: use 20.04 (Foxy) or 22.04 (Humble)"; exit 1 ;;
esac
step() { echo; echo "=== $*"; }

step "ROS 2 $DISTRO"
if [ ! -f /opt/ros/$DISTRO/setup.bash ]; then
    echo "installing ROS 2 $DISTRO (ros-base) from packages.ros.org"
    sudo apt-get update && sudo apt-get install -y curl gnupg lsb-release software-properties-common
    sudo add-apt-repository -y universe
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $UBUNTU_CODENAME main" | \
        sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt-get update && sudo apt-get install -y ros-$DISTRO-ros-base
fi

step "ROS 2 packages"
sudo apt-get update
sudo apt-get install -y ros-$DISTRO-slam-toolbox ros-$DISTRO-pointcloud-to-laserscan ros-$DISTRO-tf2-ros \
    ros-$DISTRO-rmw-cyclonedds-cpp ros-$DISTRO-rosidl-generator-dds-idl ros-$DISTRO-visualization-msgs \
    ros-$DISTRO-nav-msgs ros-$DISTRO-launch-ros python3-colcon-common-extensions python3-pip git \
    libyaml-cpp-dev libpcap-dev libeigen3-dev
sudo apt-get install -y ros-$DISTRO-realsense2-camera || echo "realsense2_camera not installed (cameras are optional)"

step "Python packages"
pip3 install --user "numpy<2" scipy onnxruntime pyyaml
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch $(python3 -c 'import torch; print(torch.__version__)') found"
else
    echo "installing PyTorch (CPU build; the map and the shield run on the CPU, ~25 ms per scan)"
    pip3 install --user torch || echo "PyTorch did not install: get NVIDIA's wheel for your JetPack (docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform)"
fi

step "CycloneDDS for the robot (Foxy only: the robot speaks CycloneDDS 0.10)"
if [ "$DISTRO" = foxy ] && [ ! -f "$HOME/cyclonedds_ws/install/setup.bash" ]; then
    mkdir -p "$HOME/cyclonedds_ws/src"
    cd "$HOME/cyclonedds_ws/src"
    [ -d rmw_cyclonedds ] || git clone https://github.com/ros2/rmw_cyclonedds -b foxy
    [ -d cyclonedds ] || git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
    cd "$HOME/cyclonedds_ws"
    # built without ROS 2 sourced, as unitree_ros2 prescribes
    env -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH colcon build --packages-select cyclonedds
    (set +u; source /opt/ros/foxy/setup.bash && colcon build --packages-select rmw_cyclonedds_cpp)
fi

step "Unitree messages and the RoboSense driver in $WS"
mkdir -p "$WS/src"
cd "$WS/src"
if [ ! -d unitree_go ]; then
    tmp=$(mktemp -d)
    git clone --depth 1 https://github.com/unitreerobotics/unitree_ros2.git "$tmp/unitree_ros2"
    cp -r "$tmp/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go" "$tmp/unitree_ros2/cyclonedds_ws/src/unitree/unitree_api" .
    rm -rf "$tmp"
fi
if [ ! -d rslidar_sdk ]; then
    git clone https://github.com/RoboSense-LiDAR/rslidar_sdk.git
    (cd rslidar_sdk && git submodule init && git submodule update)
    git clone https://github.com/RoboSense-LiDAR/rslidar_msg.git
fi
cd "$WS"
set +u
source /opt/ros/$DISTRO/setup.bash
if [ "$DISTRO" = foxy ]; then source "$HOME/cyclonedds_ws/install/setup.bash"; fi
set -u
colcon build --cmake-args -DBUILD_TESTING=OFF

step "Environment: ~/b2w_env.sh"
cat > "$HOME/b2w_env.sh" <<ENV
# B2-W navigation stack: source this in every shell (also on a laptop for RViz, with its own interface name)
source /opt/ros/$DISTRO/setup.bash
$( [ "$DISTRO" = foxy ] && echo "source $HOME/cyclonedds_ws/install/setup.bash" || true )
source $WS/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="$IFACE" priority="default" multicast="default"/></Interfaces></General></Domain></CycloneDDS>'
export B2W_REPO=$REPO
ENV
echo "wrote ~/b2w_env.sh (interface $IFACE)"

step "Check"
set +u
source "$HOME/b2w_env.sh"
set -u
python3 -c "from unitree_go.msg import SportModeState; from unitree_api.msg import Request; print('unitree messages: ok')"
PYTHONPATH="$REPO/src:$REPO/deploy" python3 "$REPO/scripts/b2w/benchmark.py" --policy "$REPO/deploy/b2w/release/policy.onnx" \
    --config "$REPO/deploy/b2w/release/deploy.json" --check "$REPO/deploy/b2w/release/samples.npz" --steps 30 || true
timeout 5 ros2 topic list 2>/dev/null | grep -E "sportmodestate|wirelesscontroller|rslidar" || \
    echo "robot topics not seen yet (robot off, or another interface/ROS_DOMAIN_ID): see deploy/b2w/README.md, step 3"
echo
echo "done. Next: source ~/b2w_env.sh, then deploy/b2w/README.md from step 3."
