import numpy as np
import mujoco 
import cv2

class RandomizeHelpers:
    def _randomize_skybox_gradient(model, _mjr_ctx):
        PALETTE = [
            np.array([0.35, 0.42, 0.32]), np.array([0.28, 0.38, 0.28]),
            np.array([0.42, 0.35, 0.48]), np.array([0.38, 0.30, 0.45]),
            np.array([0.45, 0.35, 0.28]), np.array([0.38, 0.28, 0.22]),
            np.array([0.50, 0.48, 0.46]), np.array([0.38, 0.37, 0.36]),
            np.array([0.55, 0.52, 0.48]),
        ]

        def rand_room_color():
            base = PALETTE[np.random.randint(len(PALETTE))].copy()
            base += np.random.uniform(-0.04, 0.04, size=3)
            return np.clip(base, 0.0, 1.0)

        sky_tex_id = None
        for tid in range(model.ntex):
            if int(model.tex_type[tid]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
                sky_tex_id = tid
                break

        w = int(model.tex_width[sky_tex_id])
        h = int(model.tex_height[sky_tex_id])
        c = int(model.tex_nchannel[sky_tex_id])

        canvas = np.zeros((h, w, 3), dtype=np.float32)
        num_h_blocks = np.random.randint(4, 8)
        boundaries = sorted(np.random.choice(range(1, h), size=num_h_blocks - 1, replace=False))
        boundaries = [0] + list(boundaries) + [h]

        for i in range(len(boundaries) - 1):
            y0, y1 = boundaries[i], boundaries[i + 1]
            block_color = rand_room_color()

            num_v_patches = np.random.randint(2, 5)
            col_boundaries = sorted(np.random.choice(range(1, w), size=num_v_patches - 1, replace=False))
            col_boundaries = [0] + list(col_boundaries) + [w]

            for j in range(len(col_boundaries) - 1):
                x0, x1 = col_boundaries[j], col_boundaries[j + 1]
                patch_color = np.clip(block_color + np.random.uniform(-0.07, 0.07, 3), 0.0, 1.0)
                canvas[y0:y1, x0:x1, :] = patch_color

        noise = np.random.uniform(-0.03, 0.03, size=(h, w, 3)).astype(np.float32)
        canvas = np.clip(canvas + noise, 0.0, 1.0)
        canvas_u8 = (canvas * 255).astype(np.uint8)

        for ch in range(3):
            canvas_u8[:, :, ch] = cv2.GaussianBlur(
                canvas_u8[:, :, ch], (9, 9), sigmaX=3, sigmaY=3
            )

        flat = canvas_u8.reshape(-1)
        adr = int(model.tex_adr[sky_tex_id])
        face_count = 6
        total = w * h * c * face_count

        stacked = np.tile(flat, face_count)
        model.tex_data[adr : adr + total] = stacked
        mujoco.mjr_uploadTexture(model, _mjr_ctx, sky_tex_id)
    
    @staticmethod
    def _randomize_bg_textures(tex_name: str, img_path: str, model, _mjr_ctx) -> None:
        tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_name)
        
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        w = int(model.tex_width[tex_id])
        h = int(model.tex_height[tex_id])
        c = int(model.tex_nchannel[tex_id])

        rgb_resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        if c == 3:
            img_u8 = rgb_resized.astype(np.uint8)
        else:
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            img_u8 = np.concatenate([rgb_resized.astype(np.uint8), alpha], axis=-1)

        adr = int(model.tex_adr[tex_id])
        tex_type = int(model.tex_type[tex_id])
        is_cube = (tex_type == int(mujoco.mjtTexture.mjTEXTURE_CUBE))

        face_count = 6 if is_cube else 1
        expected = w * h * c * face_count

        flat = img_u8.reshape(-1)
        if is_cube:
            stacked = np.tile(flat, face_count)
            model.tex_data[adr : adr + expected] = stacked
        else:
            model.tex_data[adr : adr + expected] = flat

        mujoco.mjr_uploadTexture(model, _mjr_ctx, tex_id)
    
    @staticmethod
    def _randomize_target_start(model, data) -> None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        rand_pos = [np.random.uniform(-0.50, -0.50), np.random.uniform(0.30,  0.90), np.random.uniform(0.2,  0.30)]
        print(rand_pos)
        model.body_pos[body_id] = rand_pos
        mujoco.mj_kinematics(model, data)