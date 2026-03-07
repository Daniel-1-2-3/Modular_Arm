"""
test_compute_action.py — always-incremental controller
Keys:  q/ESC = quit  |  r = reset  |  SPACE = pause
"""

import argparse, math, warnings
from pathlib import Path
import cv2, numpy as np
from Arm_Env.arm_env import ArmEnv
from color_detect_control import ColorDetectControl

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run(args):
    env = ArmEnv(
        xml_path=str(Path("Simulation") / "Assets" / "scene.xml"),
        width=args.img_w, height=args.img_h,
        display=False, discount=0.99,
        episode_horizon=args.episode_horizon,
        enable_domain_randomization=args.domain_rand,
    )
    ctrl = ColorDetectControl()
    step_count = 0
    paused = False

    def reset_episode():
        nonlocal step_count
        env.reset()
        ctrl.reset()
        step_count = 0

    reset_episode()
    print("=" * 60)
    print("  ctrl.step() test   q=quit  r=reset  SPACE=pause")
    print("=" * 60)

    while True:
        ctrl.run_sim(env, pov_scale=args.pov_scale, scene_div=args.scene_div)
        key = cv2.waitKey(args.delay_ms) & 0xFF
        if key in (ord("q"), 27): break
        if key == ord("r"): reset_episode(); print("\n── reset ──"); continue
        if key == ord(" "): paused = not paused; print("PAUSED" if paused else "RESUMED"); continue
        if paused: continue

        env._gl_ctx.make_current()
        image_rgb = env._render_fixedcam(env._mjv_cam_pov)
        tof_m = float(env.last_tof_m)

        # Pass env so ctrl can read joint positions for limit detection
        action = ctrl.step(image_rgb, tof_m, env=env)

        result = env.step(action)
        step_count += 1

        if step_count % 15 == 0:
            fill = ctrl._get_fill_fraction(image_rgb)
            status = "STOPPED" if ctrl.is_stopped else "active"
            mid_deg = math.degrees(ctrl._mid_pos)
            print(f"step {step_count:5d} [{status}] | tof {tof_m:.3f}m | fill {fill:.0%} | "
                  f"mid {mid_deg:.1f}° | "
                  f"act [{action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f} {action[3]:+.2f}]")

        if ctrl.is_stopped:
            print(f"\n■ STOPPED (fill >= {ctrl._get_fill_fraction(image_rgb):.0%}) "
                  f"at step {step_count}, tof={tof_m:.4f}m.  r=reset q=quit")
            while True:
                ctrl.run_sim(env, pov_scale=args.pov_scale, scene_div=args.scene_div)
                k = cv2.waitKey(100) & 0xFF
                if k == ord("r"): reset_episode(); print("── reset ──"); break
                if k in (ord("q"), 27): cv2.destroyAllWindows(); return

        st = result.get("step_type", None)
        if st is not None and hasattr(st, "value") and st.value == 2:
            print(f"\n✗ Horizon after {step_count} steps.")
            reset_episode(); print("── auto-reset ──")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--episode_horizon", type=int, default=600)
    p.add_argument("--domain_rand", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pov_scale", type=int, default=1)
    p.add_argument("--scene_div", type=int, default=2)
    p.add_argument("--delay_ms", type=int, default=30)
    run(p.parse_args())