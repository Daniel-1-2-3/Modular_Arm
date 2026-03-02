import os
import time
import math
import threading
import queue

import mujoco
import mujoco.viewer

xml_path = os.path.join("Simulation", "Assets", "scene.xml")

def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0

def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi

def clamp_to_joint_range(model: mujoco.MjModel, joint_name: str, qpos_rad: float) -> float:
    jid = model.joint(joint_name).id
    jtype = int(model.jnt_type[jid])
    if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return qpos_rad
    if int(model.jnt_limited[jid]) == 0:
        return qpos_rad
    rmin = float(model.jnt_range[jid][0])
    rmax = float(model.jnt_range[jid][1])
    return max(rmin, min(rmax, qpos_rad))

def get_joint_qpos_rad(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    jid = model.joint(joint_name).id
    qadr = int(model.jnt_qposadr[jid])
    return float(data.qpos[qadr])

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    print("Model loaded successfully!")
except ValueError as e:
    print(f"Failed to load XML: {e}")
    raise SystemExit(1)

cmd_q: "queue.Queue[str]" = queue.Queue()

def input_thread():
    print("\nCommands:")
    print("  <joint_name> <delta_degrees>   e.g.  j_b_mid +1   or   j_c_end -5")
    print("  list")
    print("  status")
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
        "(e.g., <position joint='j_b_mid' .../>)."
    )

CONTROLLED_JOINTS = sorted(joint_to_ctrl.keys())

print("\nControllable joints (from actuators):")
for jn in CONTROLLED_JOINTS:
    print(f"  {jn:20s} -> ctrl[{joint_to_ctrl[jn]}]")

targets_rad: dict[str, float] = {}
for jn in CONTROLLED_JOINTS:
    targets_rad[jn] = get_joint_qpos_rad(model, data, jn)

for jn in CONTROLLED_JOINTS:
    data.ctrl[joint_to_ctrl[jn]] = targets_rad[jn]

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat = [0, 0.6, 0.1]
    viewer.cam.distance = 1.2

    while viewer.is_running():
        step_start = time.time()

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
                print("  <joint_name> <delta_degrees>   e.g.  j_b_mid +1   or   j_c_end -5")
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
                    print(f"  {jn:20s}: target={rad_to_deg(targ):7.2f} deg | current={rad_to_deg(curr):7.2f} deg")
                continue

            parts = line.split()
            if len(parts) != 2:
                print("Bad command. Use: <joint_name> <delta_degrees> (example: j_b_mid +1). Type 'help'.")
                continue

            jn, deg_s = parts[0], parts[1]
            if jn not in joint_to_ctrl:
                print(f"Unknown joint '{jn}'. Type 'list' to see controllable joints.")
                continue

            try:
                deg_delta = float(deg_s)
            except ValueError:
                print(f"Bad delta '{deg_s}'. Example: j_b_mid +1")
                continue

            new_target = targets_rad[jn] + deg_to_rad(deg_delta)
            new_target = clamp_to_joint_range(model, jn, new_target)
            targets_rad[jn] = new_target
            print(f"Adjusted {jn} by {deg_delta:+.2f} deg -> target {rad_to_deg(new_target):.2f} deg")

        for jn in CONTROLLED_JOINTS:
            data.ctrl[joint_to_ctrl[jn]] = targets_rad[jn]

        mujoco.mj_step(model, data)
        viewer.sync()

        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)