import os, cv2, numpy as np
from Arm_Env.arm_env import ArmEnv

if __name__ == "__main__":
    env = ArmEnv(xml_path=os.path.join("Simulation", "Assets", "scene.xml"))
    drive = 1 if getattr(env, "drive_joint", None) is not None else 0
    n = len(env.controlled_joints) + drive
    km = {ord("a"):(0,-1),
          ord("d"):(0,1),
          ord("s"):(1,-1),
          ord("w"):(1,1),
          ord("f"):(2,-1),
          ord("r"):(2,1)}
    if drive: 
        km.update({ord("i"):(n-1,1), ord("k"):(n-1,-1)})

    a = np.zeros(n, np.float32)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27): break
        if k == ord("m"): env._randomize()
        a.fill(0.0)
        if k in km:
            i, s = km[k]
            a[i] = float(s)
        env.step(a)

    cv2.destroyAllWindows()