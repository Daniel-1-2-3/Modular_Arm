import os
os.environ['MUJOCO_GL'] = 'glfw' 

import time
import numpy as np
import mujoco
import cv2

class ModularArmEnv:
    def __init__(self, xml_path, width=256, height=256, camera_name="cam0"):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        # Get camera ID
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        
        self.n_frames_ran = 0
        self.last_time = time.time()

    def move_camera(self, key):
        # Access camera position and quaternion
        pos = self.model.cam_pos[self.cam_id]
        quat = self.model.cam_quat[self.cam_id]
        
        speed = 0.05
        if key == ord('w'): pos[0] -= speed # Move forward
        if key == ord('s'): pos[0] += speed # Move backward
        if key == ord('a'): pos[1] += speed # Move left
        if key == ord('d'): pos[1] -= speed # Move right
        if key == ord('r'): pos[2] += speed # Move up
        if key == ord('f'): pos[2] -= speed # Move down
        
        print(f"Cam Pos: {pos}, Cam Quat: {quat}")

    def step(self, n_substeps=10):
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

if __name__ == "__main__":
    env = ModularArmEnv(xml_path=os.path.join("Simulation", "Assets", "scene.xml"))
    
    print("Use W/A/S/D to move, R/F to move up/down. Press 'q' to quit.")
    
    while True:
        rgb = env.step(n_substeps=10)
        
        if rgb is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Viewer", cv2.flip(bgr, 0))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key != 255: # If a key was pressed
                env.move_camera(key)

    cv2.destroyAllWindows()