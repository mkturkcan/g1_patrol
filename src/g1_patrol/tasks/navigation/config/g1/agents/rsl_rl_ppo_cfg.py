from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1CrowdNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_crowd_nav"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1NycTeacherPPORunnerCfg(G1CrowdNavPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_nyc_teacher"
        self.max_iterations = 3000
        # with tanh-squashed commands, extra action noise costs little once saturated and the entropy bonus would
        # inflate it steadily: keep the bonus small
        self.algorithm.entropy_coef = 0.001
        # a 20 s horizon at 10 Hz: waiting a few seconds for a gap must not look like losing most of the episode
        self.algorithm.gamma = 0.995


@configclass
class G1CivicTeacherPPORunnerCfg(G1NycTeacherPPORunnerCfg):
    """The street teacher's recipe for the civic task (60 s episodes; a 20 s horizon at gamma 0.995)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_civic_teacher"
        self.max_iterations = 12000
        # with the trigger and brace outputs and many more events, the action noise collapsed under the street
        # teacher's entropy bonus (std 0.42 -> 0.13 in 1,000 iterations, then 0.03 for the yaw rate)
        self.algorithm.entropy_coef = 0.003


@configclass
class G1NycLidarDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distil the privileged teacher into a student that sees only the MID-360 height map (DAgger-style). The loss
    compares the tanh-squashed actions (:class:`CommandSpaceDistillation`).

    Train with ``--checkpoint <teacher model_XXXX.pt>``; the teacher network must match
    :class:`G1NycTeacherPPORunnerCfg`'s actor.
    """

    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "g1_nyc_lidar_student"
    obs_groups = {"student": ["student", "lidar_map"], "teacher": ["policy"]}
    student = RslRlCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32], kernel_size=[5, 3, 3], stride=[2, 2, 2], activation="elu", max_pool=False
        ),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="g1_patrol.tasks.navigation.config.g1.agents.distillation:CommandSpaceDistillation",
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=16,
    )


@configclass
class G1NycCameraDistillationRunnerCfg(G1NycLidarDistillationRunnerCfg):
    """Distil the teacher into a camera-only student (stacked D435i depth frames)."""

    experiment_name = "g1_nyc_camera_student"
    obs_groups = {"student": ["student", "depth", "depth_map"], "teacher": ["policy"]}


@configclass
class G1CivicLidarDistillationRunnerCfg(G1NycLidarDistillationRunnerCfg):
    """Distil the civic teacher into the lidar student (tracked people, remembered litter, MID-360 height map)."""

    experiment_name = "g1_civic_lidar_student"
    obs_groups = {"student": ["student", "lidar_map"], "teacher": ["policy"]}


@configclass
class G1CivicCameraDistillationRunnerCfg(G1NycLidarDistillationRunnerCfg):
    """Distil the civic teacher into the camera-only student."""

    experiment_name = "g1_civic_camera_student"
    obs_groups = {"student": ["student", "depth", "depth_map"], "teacher": ["policy"]}
