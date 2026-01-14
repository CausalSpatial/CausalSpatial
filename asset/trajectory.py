import torch
import os
import sys
import re
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
try:
    sys.path.append(os.path.join(project_root, "sub_module/map_anything"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
except Exception as e:
    print(f"Submodules not found!\n{e}")

from sub_module.map_anything.mapanything.utils.image import load_images
from sub_module.map_anything.mapanything.models import MapAnything

from asset.grounding import BboxExtractor




class Direction:
    def __init__(
        self,
        model,
        processor
    ):
        self.model = model
        self.processor = processor
        
        self.prompt_direction = """# Role
You are an advanced Vision-Language Model expert in physics and motion analysis.

# Context Variables
- **Target Object:** {obj}
- **Motion Context:** {mot}

# Task
Analyze the uploaded image to determine the 2D movement direction vector of the {obj}.

# Evaluation Protocol (Strict Priority)
You must evaluate the image against the following three criteria **in order**. Use the **FIRST** method that applies and ignore the rest.

**PRIORITY 1: Explicit Visual Annotations (Arrows)**
- **Check:** Is there a drawn arrow (artificial overlay) pointing to/from the {obj}?
- **Action:** If YES, the vector MUST align perfectly with this arrow. Ignore object pose.

**PRIORITY 2: Motion Artifacts (Trajectories)**
- **Check:** Is there visible motion blur, trajectory lines, or a ghosting trail attached to the {obj}?
- **Action:** If YES, follow the tangent of the trajectory at the object's current position.

**PRIORITY 3: Inherent Object Orientation (Pose)**
- **Check:** If NO arrows and NO blur exist (Static Image), analyze the geometric pose of the {obj}.
- **Action:**
    - **Determine the Longitudinal Axis:** Identify the imaginary central line running from the rear to the front of the object.
    - **Parallelism Constraint:** The direction vector MUST be **strictly parallel** to the object's side edges (e.g., parallel to the side doors or chassis line of a car).
    - **Identify "Front":** Determine which end is the front (Headlights/Grille).
    - **Construction:** Start at the Centroid. Project the vector forward along the **Longitudinal Axis**.
    - **Prohibition:** DO NOT simply connect the centroid to a specific feature like a single headlight, as this creates a diagonal skew. The line must represent the straight forward movement of the entire rigid body.

Calculate a vector defined by a Start Point and an End Point in pixel coordinates.
- **Coordinate System:** (0,0) is the top-left corner.
- **Start Point (start_x, start_y):** The geometric center (centroid) of the {obj}.
- **End Point (end_x, end_y):** A point projected outwards from the Start Point in the direction of the "Leading Edge".

# Format
Return ONLY a JSON object in the following format, with no additional text or markdown:
```json
{{
  "direction": [start_x, start_y, end_x, end_y]
}}
```"""

        self.prompt_gravity = """**Role:** Physics & Computer Vision Engine

**Task:** Analyze the input image to estimate the **3D Gravity Vector** and the **Real-to-Pixel Scale Factor**. The image depicts a sports scene (e.g., basketball, soccer) with a ball trajectory.

**Step 1: Scene & Reference Identification**
Identify the sport and select the most reliable "Standard Reference Object" for measurement:
* **Case A (Basketball):** Use the **Hoop Rim Height** (Standard: 3.05 meters from ground) or **Rim Diameter** (Standard: 0.45 meters).
* **Case B (Soccer/Football):** Use the **Goal Crossbar Height** (Standard: 2.44 meters) or the **Soccer Ball Diameter** (Standard: ~0.22 meters).
* **Case C (Generic):** Use standard Door Frames (~2.0 meters) or Fence Posts if standard height can be inferred.

**Step 2: 3D Gravity Vector Estimation**
Determine the direction of gravity $(x, y, z)$ relative to the camera's coordinate system:
* **X-axis:** Horizontal (Left/Right).
* **Y-axis:** Vertical (Up/Down in image plane).
* **Z-axis:** Depth (Into/Out of the screen).
* *Hint:* Look for vertical structures (goal posts, hoop stands, fence poles). If the camera is looking down (high pitch angle, like in the soccer image), gravity will have a significant **Z component**.
* Normalize this vector to a unit vector.
* The left top is (0, 0).

**Step 3: Scale Calculation (Real World / Pixel)**
1.  Measure the length of the chosen Reference Object in pixels ($L_{{px}}$).
2.  Retrieve its physical real-world length in meters ($L_{{real}}$).
3.  Calculate the scale factor: $S = L_{{real}} / L_{{px}}$ (Unit: meters per pixel).

**Output Format:**
Return ONLY a JSON object strictly following this structure:
```json
{
    \"scale\": 0.0162,
    \"gravity\": [x, y, z]
}
```"""


    def generate(self, prompt, image_path):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text
    
    def extract_json(self, text_output):
        try:
            pattern = r"```json\s*(\{.*?\})\s*```"
            match = re.search(pattern, text_output, re.DOTALL)
            if match: 
                return json.loads(match.group(1))
            
            pattern_loose = r"(\{.*\})"
            match_loose = re.search(pattern_loose, text_output, re.DOTALL)
            if match_loose: 
                return json.loads(match_loose.group(1))
            return None
        except Exception as e:
            return None


    def get_linear_direction(
        self,
        target_object: str,
        motion_context: str,
        image_path: str
    ):
        prompt = self.prompt_direction.format(obj=target_object, mot=motion_context)
        answer = self.generate(prompt=prompt, image_path=image_path)
        
        try:
            direction = self.extract_json(answer)["direction"]
            return direction
        except:
            return None


    def get_gravity(
        self,
        image_path
    ):
        answer = self.generate(prompt=self.prompt_gravity, image_path=image_path)
        try:
            direction = self.extract_json(answer)
            return direction
        except:
            return None



class Motion:
    def __init__(
        self,
        model,
        processor,
        g_mag: float = 9.80665,     # m/s^2
        dt: float = 1.0 / 30.0,     # s, time interval between frames
        map_anything_model: str = "facebook/map-anything"
    ):
        self.g_mag = g_mag
        self.dt = dt

        self.model = model
        self.processor = processor
        
        self.bbox_extractor = BboxExtractor(model=model, processor=processor)
        # self.mapanthing = MapAnything.from_pretrained(map_anything_model).to("cuda")
        self.mapanthing = map_anything_model
        self.direction_manager = Direction(model=model, processor=processor)


    def _get_pixel_coord(
        self, 
        image_path,
        object,
        save=False, 
        save_dir="./output"
    ):
        results = self.bbox_extractor(image_path=image_path, object=object)[0]
        x = (results["x1"] + results["x2"]) / 2.0
        y = (results["y1"] + results["y2"]) / 2.0

        if save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.bbox_extractor.draw_bbox(
                image_path=image_path,
                point_dict=results,
                output_path=os.path.join(save_dir, "bbox.png")
            )
        r = ((results["x2"] - results["x1"]) ** 2 + (results["y2"] - results["y1"]) ** 2) ** 0.5 / 2.0

        return np.array([x, y]), r


    def _map_anything(self, image_path, type=None):
        if isinstance(image_path, str):
            views = load_images([image_path])
        elif isinstance(image_path, list):
            views = load_images(image_path)

        predictions = self.mapanthing.infer(
            views,                            
            memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
            use_amp=True,                     # Use mixed precision inference (recommended)
            amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
            apply_mask=True,                  # Apply masking to dense geometry outputs
            mask_edges=True,                  # Remove edge artifacts by using normals and depth
            apply_confidence_mask=False,      # Filter low-confidence regions
            confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
        )
        pred = predictions
        if type is None:
            return pred
        else:
            return pred[type]


    def _world_to_pixel(self, pts3d_world, camera_pose, intrinsics):
        """
        pts3d_world: (N, 3) 
        camera_pose: (4, 4)
        intrinsics: (3, 3)
        return: (N, 2)
        """
        pts3d_world_ = pts3d_world.cpu().float()
        camera_pose_ = camera_pose.cpu().float()
        intrinsics_ = intrinsics.cpu().float()

        world2cam = torch.inverse(camera_pose_)

        ones = torch.ones((pts3d_world_.shape[0], 1), device=pts3d_world_.device)
        pts3d_world_h = torch.cat([pts3d_world_, ones], dim=1)  # (N, 4)

        pts3d_cam_h = (world2cam @ pts3d_world_h.T).T  # (N, 4)
        pts3d_cam = pts3d_cam_h[:, :3]

        x, y, z = pts3d_cam[:, 0], pts3d_cam[:, 1], pts3d_cam[:, 2]
        uv_h = (intrinsics_ @ torch.stack([x/z, y/z, torch.ones_like(z)], dim=0)).T  # (N, 3)

        uv = uv_h[:, :2]
        return uv


    def parabolic_motion(
        self,
        image_path_a: str,
        image_path_b: str,
        object: str,
        delta_t: float,
        frame_num: int = 60,
        save_pos_dir: str = None,
        debug: bool = False
    ):
        P_a_pixel, r = self._get_pixel_coord(image_path_a, object)
        P_b_pixel, r = self._get_pixel_coord(image_path_b, object)

        # Map anything
        pred = self._map_anything([image_path_a, image_path_b])

        # Scaling original image
        orig_w, orig_h = Image.open(image_path_a).size[:2]
        h_a, w_a = pred[0]["pts3d"].shape[1:3]
        P_a_pixel = P_a_pixel * np.array([w_a / orig_w, h_a / orig_h])
        P_b_pixel = P_b_pixel * np.array([w_a / orig_w, h_a / orig_h])

        # Get 3D points
        P_a = pred[0]["pts3d"][0, int(P_a_pixel[1]), int(P_a_pixel[0])].cpu().numpy()
        P_b = pred[1]["pts3d"][0, int(P_b_pixel[1]), int(P_b_pixel[0])].cpu().numpy()
        
        # Gravity vector
        metric_scaling_factor = pred[0]["metric_scaling_factor"][0].cpu().item()
        # g = np.array([0, self.g_mag / metric_scaling_factor, 0])  # m/s^2
        g = np.array([0, self.g_mag, 0])  # m/s^2

        # Initial velocity
        v0 = (P_b - P_a) / delta_t  # m/s
        ts = np.arange(0, frame_num + 1, dtype=np.float32) * self.dt

        # Compute trajectory
        pos = P_a + v0[None, :] * ts[:, None] + 0.5 * g[None, :] * (ts[:, None] ** 2)

        
        K = pred[0]["intrinsics"][0]             # (3, 3)
        pose = pred[0]["camera_poses"][0]        # (4, 4)
        pixels = self._world_to_pixel(pos, pose, K)
        pixels = pixels.numpy() * np.array([orig_w / w_a, orig_h / h_a])

        if debug:
            self.draw_points_on_image(
                image_path_a, pixels, 
                save_path=os.path.join(save_pos_dir, "trajectory.png" if save_pos_dir is not None else "trajectory.png"),
            )

        return pixels, r


    def linear_motion(
        self,
        image_path_a: str,
        image_path_b: str,
        object: str,
        motion_context: str,
        delta_t: float,
        frame_num: int = 60,
        save_pos_dir: str = None,
        debug: bool = False
    ):
        P_a_pixel, r = self._get_pixel_coord(image_path_a, object)
        P_b_pixel, r = self._get_pixel_coord(image_path_b, object)

        # Map anything
        pred = self._map_anything([image_path_a, image_path_b])

        # Scaling original image
        orig_w, orig_h = Image.open(image_path_a).size[:2]
        h_a, w_a = pred[0]["pts3d"].shape[1:3]
        P_a_pixel = P_a_pixel * np.array([w_a / orig_w, h_a / orig_h])
        P_b_pixel = P_b_pixel * np.array([w_a / orig_w, h_a / orig_h])

        # Get 3D points
        P_a = pred[0]["pts3d"][0, int(P_a_pixel[1]), int(P_a_pixel[0])].cpu().numpy()
        P_b = pred[1]["pts3d"][0, int(P_b_pixel[1]), int(P_b_pixel[0])].cpu().numpy()

        # Initial velocity
        v0 = (P_b - P_a) / delta_t  # m/s
        ts = np.arange(0, frame_num + 1, dtype=np.float32) * self.dt

        # Compute trajectory
        pos = P_a + v0[None, :] * ts[:, None]

        if save_pos_dir is not None:
            if not os.path.exists(save_pos_dir):
                os.makedirs(save_pos_dir)
            np.save(os.path.join(save_pos_dir, "position_pred.npy"), pos)

        K = pred[0]["intrinsics"][0]             # (3, 3)
        pose = pred[0]["camera_poses"][0]        # (4, 4)
        pixels = self._world_to_pixel(torch.tensor(pos), pose, K)
        pixels = pixels.cpu().numpy() * np.array([orig_w / w_a, orig_h / h_a])
        
        if debug:
            self.draw_points_on_image(
                image_path_a, pixels, 
                save_path=os.path.join(save_pos_dir, "trajectory.png" if save_pos_dir is not None else "trajectory.png"),
            )

        return pixels, r

    def linear_motion_v3(
        self,
        image_path_a: str,
        image_path_b: str,
        object: str,
        motion_context: str,
        delta_t: float,
        frame_num: int = 60,
        save_pos_dir: str = None,
        debug: bool = False
    ):
        P_a_pixel, r = self._get_pixel_coord(image_path_a, object)
        P_b_pixel, r = self._get_pixel_coord(image_path_b, object)

        velocity_pixel = (P_b_pixel - P_a_pixel) / delta_t
        ts = np.arange(0, frame_num + 1, dtype=np.float32) * self.dt

        pixels = P_a_pixel[None, :] + velocity_pixel[None, :] * ts[:, None]

        if save_pos_dir is not None:
            if not os.path.exists(save_pos_dir):
                os.makedirs(save_pos_dir)
            np.save(os.path.join(save_pos_dir, "position_pred_pixel.npy"), pixels)

        if debug:
            self.draw_points_on_image(
                image_path_a, pixels, 
                save_path=os.path.join(
                    save_pos_dir, 
                    "trajectory_pixel.png" if save_pos_dir is not None else "trajectory_pixel.png"
                ))
        return pixels, r


    def linear_motion_single(
        self,
        image_path: str,
        object: str,
        motion_context: str,
        velocity_abs: int = 3,
        frame_num: int = 60,
        save_pos_dir: str = None,
        debug: bool = False
    ):
        # calculate the pixel coordinate
        P_pixel, r = self._get_pixel_coord(image_path, object)

        # calculate the veclocity
        velocity_dir = self.direction_manager.get_linear_direction(
            object, motion_context, image_path)
        velocity_pixel = np.array([
            velocity_dir[2] - velocity_dir[0],
            velocity_dir[3] - velocity_dir[1]
        ], dtype=np.float32)
        norm = np.linalg.norm(velocity_pixel)
        velocity_pixel = velocity_pixel / norm * velocity_abs

        ts = np.arange(0, frame_num + 1, dtype=np.float32) * self.dt

        pixels = P_pixel[None, :] + velocity_pixel[None, :] * ts[:, None]

        if save_pos_dir is not None:
            if not os.path.exists(save_pos_dir):
                os.makedirs(save_pos_dir)
            np.save(os.path.join(save_pos_dir, "position_pred_pixel.npy"), pixels)

        if debug:
            self.draw_points_on_image(
                image_path, pixels, 
                save_path=os.path.join(
                    save_pos_dir, 
                    "trajectory_pixel.png" if save_pos_dir is not None else "trajectory_pixel.png"
                ))
        return pixels, r


    def parabolic_motion_single(
        self,
        image_path: str,
        object: str,
        motion_context: str,
        velocity_abs: int = 30,
        frame_num: int = 60,
        save_pos_dir: str = None,
        debug: bool = False
    ):
        # calculate the pixel coordinate
        P_pixel, r = self._get_pixel_coord(image_path, object)

        # calculate the veclocity
        velocity_dir = self.direction_manager.get_linear_direction(
            object, motion_context, image_path)
        velocity_pixel = np.array([
            velocity_dir[2] - velocity_dir[0],
            velocity_dir[3] - velocity_dir[1]
        ], dtype=np.float32)
        norm = np.linalg.norm(velocity_pixel)
        velocity_pixel = velocity_pixel / norm * velocity_abs

        # calculate the gravity
        gravity = self.direction_manager.get_gravity(image_path)
        gravity_vec = np.array(gravity["gravity"], dtype=np.float32)
        norm = np.linalg.norm(gravity_vec)
        # gravity_vec = gravity_vec / norm * gravity["scale"] * 980
        gravity_vec = gravity_vec / norm * 98
        gravity_vec = gravity_vec[:-1]

        ts = np.arange(0, frame_num + 1, dtype=np.float32) * self.dt

        pixels = P_pixel[None, :] + velocity_pixel[None, :] * ts[:, None] + 0.5 * gravity_vec[None, :] * (ts[:, None] ** 2)

        if save_pos_dir is not None:
            if not os.path.exists(save_pos_dir):
                os.makedirs(save_pos_dir)
            np.save(os.path.join(save_pos_dir, "position_pred_pixel.npy"), pixels)

        if debug:
            self.draw_points_on_image(
                image_path, pixels, 
                save_path=os.path.join(
                    save_pos_dir, 
                    "trajectory_pixel.png" if save_pos_dir is not None else "trajectory_pixel.png"
                ))
        return pixels, r


    def draw_points_on_image(self, image, points, save_path, r=10, color='lime', fill=True, alpha=0.8):
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        points = np.array(points)
        xs = points[:, 0]
        ys = points[:, 1]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        for (x, y) in zip(xs, ys):
            circle = patches.Circle(
                (x, y),
                radius=r,
                edgecolor=color,
                facecolor=color if fill else 'none',
                linewidth=1.5,
                alpha=alpha
            )
            ax.add_patch(circle)
        
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)



