import os
import time
import math
import threading
import queue

import mujoco
import mujoco.viewer

# Absolute-ish path (relative to your current working directory)
xml_path = os.path.join("Simulation", "Assets", "scene.xml")

# ----------------------------
# Helpers
# ----------------------------
def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0

def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi

def clamp_to_joint_range(model: mujoco.MjModel, joint_name: str, qpos_rad: float) -> float:
    jid = model.joint(joint_name).id
    jtype = int(model.jnt_type[jid])

    # Only clamp hinges/slides
    if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return qpos_rad

    # If joint isn't limited, don't clamp
    if int(model.jnt_limited[jid]) == 0:
        return qpos_rad

    rmin = float(model.jnt_range[jid][0])
    rmax = float(model.jnt_range[jid][1])
    return max(rmin, min(rmax, qpos_rad))

def get_joint_qpos_rad(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    jid = model.joint(joint_name).id
    qadr = int(model.jnt_qposadr[jid])
    return float(data.qpos[qadr])

# ----------------------------
# Load model
# ----------------------------
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print("Model loaded successfully!")
except ValueError as e:
    print(f"Failed to load XML: {e}")
    raise SystemExit(1)

# ----------------------------
# Terminal command interface
# ----------------------------
cmd_q: "queue.Queue[str]" = queue.Queue()

def input_thread():
    print("\nCommands:")
    print("  <joint_name> <degrees>   e.g.  j_mid 45")
    print("  list                     list controllable joints")
    print("  status                   show target + current (deg)")
    print("  help")
    print("  quit / q")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            cmd_q.put("quit")
            return
        cmd_q.put(line)

threading.Thread(target=input_thread, daemon=True).start()

# ----------------------------
# Joint -> actuator mapping
# ----------------------------
# Map joint name -> ctrl index, by reading actuator transmissions.
# This assumes joint actuators exist (like <position joint="...">).
joint_to_ctrl: dict[str, int] = {}
for act_id in range(model.nu):
    trntype = int(model.actuator_trntype[act_id])
    if trntype != int(mujoco.mjtTrn.mjTRN_JOINT):
        continue

    j_id = int(model.actuator_trnid[act_id][0])
    if 0 <= j_id < model.njnt:
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        if j_name:
            joint_to_ctrl[j_name] = act_id

if not joint_to_ctrl:
    raise RuntimeError(
        "No JOINT actuators found. Your XML must have actuators that target joints "
        "(e.g., <position joint='j_mid' .../>)."
    )

# Make control order stable for printing / iteration
CONTROLLED_JOINTS = sorted(joint_to_ctrl.keys())

print("\nControllable joints (from actuators):")
for jn in CONTROLLED_JOINTS:
    print(f"  {jn:20s} -> ctrl[{joint_to_ctrl[jn]}]")

# ----------------------------
# Targets: "hold position" behavior
# ----------------------------
# Hold targets for every joint. When you command one joint, we "brace" the others by
# freezing them at their CURRENT angle immediately. This prevents the chain from
# dragging other joints off their targets as easily.
targets_rad: dict[str, float] = {}

# Initialize targets to current pose
for jn in CONTROLLED_JOINTS:
    targets_rad[jn] = get_joint_qpos_rad(model, data, jn)

# Write initial ctrl (holds current pose)
for jn in CONTROLLED_JOINTS:
    data.ctrl[joint_to_ctrl[jn]] = targets_rad[jn]

# ----------------------------
# Viewer + sim loop
# ----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat = [0, 0.6, 0.1]
    viewer.cam.distance = 1.2

    while viewer.is_running():
        step_start = time.time()

        # Process all pending terminal commands (non-blocking)
        while True:
            try:
                line = cmd_q.get_nowait()
            except queue.Empty:
                break

            if not line:
                continue

            low = line.lower()
            if low in ("q", "quit", "exit"):
                print("Exiting.")
                raise SystemExit(0)

            if low == "help":
                print("  <joint_name> <degrees>   e.g.  j_mid 45")
                print("  list")
                print("  status")
                print("  quit / q")
                continue

            if low == "list":
                for jn in CONTROLLED_JOINTS:
                    print(f"  {jn:20s} -> ctrl[{joint_to_ctrl[jn]}]")
                continue

            if low == "status":
                for jn in CONTROLLED_JOINTS:
                    curr = get_joint_qpos_rad(model, data, jn)
                    targ = targets_rad[jn]
                    print(
                        f"  {jn:20s}: target={rad_to_deg(targ):7.2f} deg | current={rad_to_deg(curr):7.2f} deg"
                    )
                continue

            # Expect: joint_name degrees
            parts = line.split()
            if len(parts) != 2:
                print("Bad command. Use: <joint_name> <degrees> (example: j_mid 45). Type 'help'.")
                continue

            jn, deg_s = parts[0], parts[1]
            if jn not in joint_to_ctrl:
                print(f"Unknown joint '{jn}'. Type 'list' to see controllable joints.")
                continue

            try:
                deg_val = float(deg_s)
            except ValueError:
                print(f"Bad degrees '{deg_s}'. Example: j_mid 45")
                continue

            # ---- HOLD / STALL FEEL ----
            # Brace all other joints at their current angles right now.
            for other in CONTROLLED_JOINTS:
                if other == jn:
                    continue
                targets_rad[other] = clamp_to_joint_range(model, other, get_joint_qpos_rad(model, data, other))

            # Set commanded joint target
            rad_val = clamp_to_joint_range(model, jn, deg_to_rad(deg_val))
            targets_rad[jn] = rad_val
            print(f"Set {jn} target to {rad_to_deg(rad_val):.2f} deg")

        # Hold joints at targets every step
        for jn in CONTROLLED_JOINTS:
            data.ctrl[joint_to_ctrl[jn]] = targets_rad[jn]

        mujoco.mj_step(model, data)
        viewer.sync()

        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)