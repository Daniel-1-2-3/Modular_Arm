import os
from dm_env import StepType
from Arm_Env.arm_env import ArmEnv
import cv2

if __name__ == "__main__":
    xml_path = os.path.join("Simulation", "Assets", "scene.xml")
    arm_env = ArmEnv(xml_path=xml_path, display=True, episode_horizon=300)

    drive_dims = 1 if getattr(arm_env, "drive_joint", None) is not None else 0
    action_len = len(arm_env.controlled_joints) + drive_dims

    print(f"Action space:      {arm_env.get_action_space()}")
    print(f"Observation space: {arm_env.get_observation_space()}")
    print(f"Action dims:       {action_len}  (joints: {arm_env.controlled_joints} + drive: {arm_env.drive_joint})")

    for episode in range(3):
        time_step = arm_env.reset()
        print(f"\n Episode {episode + 1}")
        print(f" reset() keys:     {list(time_step.keys())}")
        print(f" obs shape:        {time_step["pov_obs"].shape}")
        print(f" tof obs:          {time_step["tof_obs"]}")

        episode_reward = 0
        step = 0

        while time_step["step_type"] != StepType.LAST:
            action = arm_env.get_action_space().sample()

            time_step = arm_env.step(action)
            episode_reward += time_step["reward"]
            step += 1
            
            if step % 50 == 0:
                print(f" step {step:>3} | reward: {time_step['reward']:.4f} | "
                      f"tof: {time_step['tof_obs']:.4f} m | "
                      f"tof_hit: {arm_env._tof_hitting_target()} | "
                      f"h: {arm_env.last_align_h_deg:.1f} deg | "
                      f"v_err: {arm_env.last_align_v_deg - arm_env.last_opt_v_deg:.1f} deg")

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                cv2.destroyAllWindows()
                raise SystemExit(0)

    cv2.destroyAllWindows()