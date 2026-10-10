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


@configclass
class G1NycLidarDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distil the privileged teacher into a student that sees only the MID-360 height map (DAgger-style).

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
    algorithm = RslRlDistillationAlgorithmCfg(num_learning_epochs=2, learning_rate=5.0e-4, gradient_length=16)


@configclass
class G1NycCameraDistillationRunnerCfg(G1NycLidarDistillationRunnerCfg):
    """Distil the teacher into a camera-only student (stacked D435i depth frames)."""

    experiment_name = "g1_nyc_camera_student"
    obs_groups = {"student": ["student", "depth"], "teacher": ["policy"]}
