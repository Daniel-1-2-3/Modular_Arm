import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("MUJOCO_PLATFORM", None)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from pathlib import Path
import numpy as np
import torch
import time
import argparse
from dm_env import specs
from gymnasium.spaces import Box

import DrQv2_Architecture.utils as utils
from DrQv2_Architecture.logger import Logger
from DrQv2_Architecture.replay_buffer import ReplayBufferStorage, make_replay_loader
from DrQv2_Architecture.video import VideoRecorder
from DrQv2_Architecture.drqv2 import DrQV2Agent

from DrQv2_Architecture.env_wrappers import (
    FrameStackWrapper,
    ActionRepeatWrapper,
    ExtendedTimeStepWrapper,
)

from Arm_Env.arm_env import ArmEnv
torch.backends.cudnn.benchmark = True

class Workshop:
    def __init__(
        self,
        # General
        device: torch.device | None = None,

        # RL
        buffer_size: int = 100_000,
        total_timesteps: int = 500_000,
        learning_starts: int = 10_000,
        num_expl_steps: int = 5_000,
        episode_horizon: int = 300,
        batch_size: int = 64,
        critic_target_tau: float = 0.001,
        update_every_steps: int = 2,
        update_mvmae_every_steps: int = 10,
        stddev_schedule: str = 'linear(1.0,0.1,500000)',
        stddev_clip: float = 0.3,
        lr: float = 1e-4,
        discount: float = 0.99,
        action_repeat: int = 2,

        # MVMAE
        mvmae_patch_size: int = 8,
        mvmae_encoder_embed_dim: int = 256,
        mvmae_decoder_embed_dim: int = 128,
        mvmae_encoder_heads: int = 16,
        mvmae_decoder_heads: int = 16,
        masking_ratio: float = 0.75,
        coef_mvmae: float = 0.005,

        # Actor+Critic
        feature_dim: int = 100,
        hidden_dim: int = 1024,

        # Image
        xml_path: str = str(Path("Simulation") / "Assets" / "scene.xml"),
        img_h_size: int = 64,
        img_w_size: int = 64,
        num_frames: int = 3,   # frame stack depth; in_channels = 3*num_frames
        display: bool = False, # ArmEnv handles display internally; keep False for training speed
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.buffer_size = buffer_size
        self.total_timesteps = total_timesteps
        self.learning_starts = learning_starts
        self.num_expl_steps = num_expl_steps
        self.episode_horizon = episode_horizon
        self.batch_size = batch_size
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.update_mvmae_every_steps = update_mvmae_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.lr = lr
        self.discount = discount
        self.action_repeat = action_repeat

        self.mvmae_patch_size = mvmae_patch_size
        self.mvmae_encoder_embed_dim = mvmae_encoder_embed_dim
        self.mvmae_decoder_embed_dim = mvmae_decoder_embed_dim
        self.mvmae_encoder_heads = mvmae_encoder_heads
        self.mvmae_decoder_heads = mvmae_decoder_heads
        self.masking_ratio = masking_ratio
        self.coef_mvmae = coef_mvmae

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.xml_path = xml_path
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size
        self.num_frames = num_frames
        self.in_channels = 3 * num_frames
        self.display = display

        self.work_dir = Path.cwd()
        print(f'Workspace: {self.work_dir}')
        self._global_step = 0
        self._global_episode = 0

        self.seed = 1
        utils.set_seed_everywhere(self.seed)

        self.train_env = self.make_env(display=self.display)
        self.eval_env = self.make_env(display=False)  # keep eval fast; videos still possible via VideoRecorder if supported

        self.action_dim = int(self.train_env.action_dim)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self.setup_replay()
        self.agent = self.make_agent()

        self.timer = utils.Timer()
        self.eval_every_frames = 10_000
        self.num_eval_episodes = 10
        self.video_recorder = VideoRecorder(self.work_dir / "Training_Results")

    def make_agent(self):
        return DrQV2Agent(
            action_shape=(self.action_space.shape[0],),
            device=self.device,
            lr=self.lr,
            logger=self.logger,

            mvmae_patch_size=self.mvmae_patch_size,
            mvmae_encoder_embed_dim=self.mvmae_encoder_embed_dim,
            mvmae_decoder_embed_dim=self.mvmae_decoder_embed_dim,
            mvmae_encoder_heads=self.mvmae_encoder_heads,
            mvmae_decoder_heads=self.mvmae_decoder_heads,

            in_channels=self.in_channels,
            img_h_size=self.img_h_size,
            img_w_size=self.img_w_size,
            masking_ratio=self.masking_ratio,
            coef_mvmae=self.coef_mvmae,

            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,

            critic_target_tau=self.critic_target_tau,
            num_expl_steps=self.num_expl_steps,
            update_every_steps=self.update_every_steps,
            update_mvmae_every_steps=self.update_mvmae_every_steps,
            stddev_schedule=self.stddev_schedule,
            stddev_clip=self.stddev_clip,
        )

    def make_env(self, display: bool):
        env = ArmEnv(
            xml_path=self.xml_path,
            width=self.img_w_size,
            height=self.img_h_size,
            display=display,
            discount=self.discount,
            episode_horizon=self.episode_horizon,
        )
        env = FrameStackWrapper(env, num_frames=self.num_frames)
        env = ActionRepeatWrapper(env, num_repeats=self.action_repeat)
        env = ExtendedTimeStepWrapper(env)
        return env

    def setup_replay(self):
        self.logger = Logger(self.work_dir / "Training_Results")

        # pov is (C,H,W) uint8 where C = 3*num_frames (normalize on GPU later)
        # tof is (1,) float32
        data_specs = (
            specs.Array((self.in_channels, self.img_h_size, self.img_w_size), np.uint8, name="pov"),
            specs.Array((1,), np.float32, name="tof"),
            specs.Array(self.action_space.shape, self.action_space.dtype, name="action"),
            specs.Array((1,), np.float32, name="reward"),
            specs.Array((1,), np.float32, name="discount"),
        )

        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / 'Buffer')
        self.replay_loader = make_replay_loader(
            self.work_dir / 'Buffer',
            self.buffer_size,
            self.batch_size,
            num_workers=4,
            save_snapshot=False,
            nstep=3,
            discount=self.discount,
        )
        self._replay_iter = None

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _ts_to_obs_dict(self, time_step):
        return {"pov": time_step.pov, "tof": time_step.tof}

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            ep_reward = 0.0
            ep_len = 0
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(self._ts_to_obs_dict(time_step), self.global_step, eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                r = float(time_step.reward[0])
                total_reward += r
                ep_reward += r
                step += 1
                ep_len += 1

            self.logger.log(
                self.global_step,
                frame=self.global_frame,
                eval_episode=episode,
                eval_episode_reward=ep_reward,
                eval_episode_length=ep_len * self.action_repeat,
                episode=self.global_episode,
            )

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        self.logger.log(
            self.global_step,
            frame=self.global_frame,
            eval_episode_reward_mean=(total_reward / episode),
            eval_episode_length_mean=(step * self.action_repeat / episode),
            episode=self.global_episode,
        )

    def train(self):
        train_until_step = utils.Until(self.total_timesteps, self.action_repeat)
        seed_until_step = utils.Until(self.learning_starts, self.action_repeat)
        eval_every_step = utils.Every(self.eval_every_frames, self.action_repeat)

        episode_step, episode_reward = 0, 0.0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)

        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.action_repeat
                    self.logger.log(
                        self.global_step,
                        frame=self.global_frame,
                        fps=(episode_frame / elapsed_time),
                        total_time=total_time,
                        episode_reward=episode_reward,
                        episode_length=episode_frame,
                        episode=self.global_episode,
                        buffer_size=len(self.replay_storage),
                    )

                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                episode_step, episode_reward = 0, 0.0

            if eval_every_step(self.global_step):
                self.logger.log(self.global_frame, eval_total_time=self.timer.total_time())
                self.eval()

            t0 = time.perf_counter()
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(self._ts_to_obs_dict(time_step), self.global_step, eval_mode=False)

            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)

            time_step = self.train_env.step(action)
            episode_reward += float(time_step.reward[0])
            self.replay_storage.add(time_step)

            episode_step += 1
            self._global_step += 1

def save_agent(agent: DrQV2Agent, path="agent_weights.pt", step: int | None = None):
    torch.save({
        "step": step,
        "mvmae": agent.mvmae.state_dict(),
        "trunc": agent.trunc.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "trunc_target": agent.trunc_target.state_dict(),
        "actor_optim": agent.actor_optim.state_dict(),
        "critic_optim": agent.critic_optim.state_dict(),
        "mvmae_optim": agent.mvmae_optim.state_dict(),
        "cfg": {
            "in_channels": agent.in_channels,
            "img_h_size": agent.img_h_size,
            "img_w_size": agent.img_w_size,
            "patch_size": agent.mvmae_patch_size,
            "encoder_embed_dim": agent.mvmae_encoder_embed_dim,
            "decoder_embed_dim": agent.mvmae_decoder_embed_dim,
            "encoder_heads": agent.mvmae_encoder_heads,        
            "decoder_heads": agent.mvmae_decoder_heads,        
            "feature_dim": agent.feature_dim,
            "hidden_dim": agent.hidden_dim,
            "masking_ratio": agent.masking_ratio,
            "action_shape": tuple(agent.action_shape),
        }
    }, path)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--num_expl_steps", type=int, default=5000)
    parser.add_argument("--episode_horizon", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--critic_target_tau", type=float, default=0.001)
    parser.add_argument("--update_every_steps", type=int, default=2)
    parser.add_argument("--update_mvmae_every_steps", type=int, default=10)
    parser.add_argument("--stddev_schedule", type=str, default="linear(1.0,0.1,500000)")
    parser.add_argument("--stddev_clip", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--mvmae_patch_size", type=int, default=8)
    parser.add_argument("--mvmae_encoder_embed_dim", type=int, default=256)
    parser.add_argument("--mvmae_decoder_embed_dim", type=int, default=128)
    parser.add_argument("--mvmae_encoder_heads", type=int, default=16)
    parser.add_argument("--mvmae_decoder_heads", type=int, default=16)
    parser.add_argument("--masking_ratio", type=float, default=0.75)
    parser.add_argument("--coef_mvmae", type=float, default=0.005)
    parser.add_argument("--feature_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--xml_path", type=str, default=str(Path("Simulation") / "Assets" / "scene.xml"))
    parser.add_argument("--img_h_size", type=int, default=64)
    parser.add_argument("--img_w_size", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--display", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dev = None if args.device is None else torch.device(args.device)
    kwargs = vars(args)
    kwargs["device"] = dev
    workspace = Workshop(**kwargs)
    workspace.train()
    save_agent(workspace.agent)