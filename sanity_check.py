"""
sanity_check.py
───────────────
Interactive joint-direction tester.

Commands (type in terminal, press Enter):
    base <deg>       rotate base by X degrees   (e.g. "base 10", "base -10")
    mid <deg>        rotate mid joint by X deg   (e.g. "mid 5",  "mid -5")
    end <deg>        rotate end effector by X deg (e.g. "end 20", "end -20")
    drive <cm>       move rail by X cm           (e.g. "drive 5", "drive -5")
    reset            reset episode
    quit / q         exit

Each command applies the delta in small increments over ~0.5s so you can
watch the motion.  The script prints the ctrl vector before and after.
"""

import os
import sys
import math
import threading
import numpy as np
import cv2
from Arm_Env.arm_env import ArmEnv


# ──────────────────────────────────────────────────────────────────────────────

def build_env():
    return ArmEnv(
        xml_path=os.path.join("Simulation", "Assets", "scene.xml"),
        display=False,
        width=640,
        height=480,
    )


SCENE_WINDOW = "scene"
POV_WINDOW = "pov"


def show(env: ArmEnv):
    env._gl_ctx.make_current()
    scene_bgr = cv2.cvtColor(env._render_fixedcam(env._mjv_cam_scene), cv2.COLOR_RGB2BGR)
    env._draw_hud(scene_bgr)
    cv2.imshow(SCENE_WINDOW, scene_bgr)

    raw = env._render_fixedcam(env._mjv_cam_pov)
    cv2.imshow(POV_WINDOW, cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))


# ── The four joint functions ──────────────────────────────────────────────────

def _apply_delta(env, ctrl, joint_idx, delta, steps=30, delay_ms=16):
    """
    Incrementally add `delta` to ctrl[joint_idx] over `steps` simulation
    steps, rendering each frame so the motion is visible.
    """
    per_step = delta / steps
    for _ in range(steps):
        ctrl[joint_idx] += per_step
        env.step(ctrl.copy())
        show(env)
        cv2.waitKey(delay_ms)


def base_rotate(env, ctrl, degrees):
    """Rotate the base joint by `degrees`. Positive = ???  (you decide)."""
    radians = math.radians(-degrees)
    print(f"  base_rotate({degrees:+.1f} deg = {radians:+.4f} rad)")
    print(f"    ctrl BEFORE: {np.array2string(ctrl, precision=4)}")
    _apply_delta(env, ctrl, joint_idx=0, delta=radians)
    print(f"    ctrl AFTER : {np.array2string(ctrl, precision=4)}")


def mid_rotate(env, ctrl, degrees):
    """Rotate the mid (elbow) joint by `degrees`."""
    radians = math.radians(-degrees)
    print(f"  mid_rotate({degrees:+.1f} deg = {radians:+.4f} rad)")
    print(f"    ctrl BEFORE: {np.array2string(ctrl, precision=4)}")
    _apply_delta(env, ctrl, joint_idx=1, delta=radians)
    print(f"    ctrl AFTER : {np.array2string(ctrl, precision=4)}")


def end_rotate(env, ctrl, degrees):
    """Rotate the end-effector (wrist) joint by `degrees`."""
    radians = math.radians(-degrees)
    print(f"  end_rotate({degrees:+.1f} deg = {radians:+.4f} rad)")
    print(f"    ctrl BEFORE: {np.array2string(ctrl, precision=4)}")
    _apply_delta(env, ctrl, joint_idx=2, delta=radians)
    print(f"    ctrl AFTER : {np.array2string(ctrl, precision=4)}")


def drive_move(env, ctrl, cm):
    """Move the linear rail by `cm` centimetres (converted to metres)."""
    metres =  - cm / 100.0
    print(f"  drive_move({cm:+.1f} cm = {metres:+.4f} m)")
    print(f"    ctrl BEFORE: {np.array2string(ctrl, precision=4)}")
    _apply_delta(env, ctrl, joint_idx=3, delta=metres)
    print(f"    ctrl AFTER : {np.array2string(ctrl, precision=4)}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    env = build_env()
    cv2.namedWindow(SCENE_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(POV_WINDOW, cv2.WINDOW_AUTOSIZE)

    # ctrl tracks the current absolute joint targets (radians / metres)
    ctrl = np.zeros(4, dtype=np.float64)

    env.reset()
    show(env)
    cv2.waitKey(1)

    print("=" * 60)
    print("  Joint direction sanity checker")
    print("  Commands:  base <deg>  mid <deg>  end <deg>  drive <cm>")
    print("             reset      quit")
    print("=" * 60)

    # We need to pump cv2.waitKey on the main thread to keep windows alive,
    # so read stdin in a background thread.
    cmd_queue = []
    stop_flag = threading.Event()

    def stdin_reader():
        while not stop_flag.is_set():
            try:
                line = input(">>> ").strip()
                if line:
                    cmd_queue.append(line)
            except EOFError:
                cmd_queue.append("quit")
                break

    t = threading.Thread(target=stdin_reader, daemon=True)
    t.start()

    while True:
        # Keep GUI responsive
        cv2.waitKey(30)
        show(env)

        if not cmd_queue:
            continue

        line = cmd_queue.pop(0)
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "q"):
            break

        if cmd == "reset":
            env.reset()
            ctrl[:] = 0.0
            print("  ── reset ──")
            show(env)
            continue

        if len(parts) != 2:
            print(f"  Usage: {cmd} <value>")
            continue

        try:
            val = float(parts[1])
        except ValueError:
            print(f"  Bad number: {parts[1]}")
            continue

        if cmd == "base":
            base_rotate(env, ctrl, val)
        elif cmd == "mid":
            mid_rotate(env, ctrl, val)
        elif cmd == "end":
            end_rotate(env, ctrl, val)
        elif cmd == "drive":
            drive_move(env, ctrl, val)
        else:
            print(f"  Unknown command: {cmd}")
            print("  Try: base, mid, end, drive, reset, quit")

    stop_flag.set()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()