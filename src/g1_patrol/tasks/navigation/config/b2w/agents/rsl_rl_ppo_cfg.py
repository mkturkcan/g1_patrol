"""RSL-RL configurations for the B2-W navigation teacher and its lidar student."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
)

from ...g1.agents.rsl_rl_ppo_cfg import TargetNoisePpoAlgorithmCfg


@configclass
class B2WTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """The street teacher's recipe (G1), with the action noise held at 0.35 (:class:`TargetNoisePPO`)."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "b2w_nav_teacher"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = TargetNoisePpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        target_std=0.35,
    )


@configclass
class B2WLidarDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distil the teacher into a student that sees the lidar height map and odometry (DAgger-style); the loss compares
    tanh-squashed actions. The student is small enough for a Jetson Orin NX (one 80 x 80 map, three conv layers).

    Train with ``--checkpoint <teacher model_XXXX.pt>``; the teacher network must match
    :class:`B2WTeacherPPORunnerCfg`'s actor.
    """

    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "b2w_nav_lidar_student"
    obs_groups = {"student": ["student", "lidar_map"], "teacher": ["policy"]}
    student = RslRlCNNModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32, 32], kernel_size=[5, 3, 3, 3], stride=[2, 2, 2, 2], activation="elu", max_pool=False
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
class B2WFusedDistillationRunnerCfg(B2WLidarDistillationRunnerCfg):
    """Distil the teacher into the student with the fused lidar and depth-camera height map."""

    experiment_name = "b2w_nav_fused_student"
    obs_groups = {"student": ["student", "height_map"], "teacher": ["policy"]}
