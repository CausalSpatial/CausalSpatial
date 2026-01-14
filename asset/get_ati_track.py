import io
import torch
import os

import numpy as np

from PIL import Image, ImageDraw

from asset.trajectory import Motion



class AtiTrackManager:
    def __init__(
        self, 
        quant_multi=8
    ):
        # ATI default
        self.quant_multi = quant_multi
        self.fixed_length = 121


    def save(self, track, output_path, compressed=True):
        bio = io.BytesIO()
        if compressed:
            np.savez_compressed(bio, array=track)
        else:
            np.savez(bio, array=track)
        torch.save(bio.getvalue(), output_path)


    def load(self, path):
        raw = torch.load(path)
        bio = io.BytesIO(raw)
        with np.load(bio) as npz:
            return npz['array']


    def _pad_points(self, points):
        """Convert points of [N, 3] to fixed length [fixed_length, 1, 3] by padding zeros or truncating."""
        points_ = points.copy()
        points_[:, -1] = 1
        n = points_.shape[0]
        if n < self.fixed_length:
            pad = np.zeros((self.fixed_length - n, 3), dtype=points_.dtype)
            points_ = np.vstack((points_, pad))
        else:
            points_ = points_[:self.fixed_length]
        return points_.reshape(self.fixed_length, 1, 3)
    
    
    def _transfer_to_static_array(self, point):
        """Set a static point [fixed_length, 1, 3] from input point (x, y)."""
        point_ = np.array([point[0], point[1], 1], dtype=np.float32).reshape(1, 3)
        points_ = np.repeat(point_[None, :], self.fixed_length, axis=0)
        return points_
        

    def _sample_static_points(
        self,
        image: Image.Image | np.ndarray, 
        traj: np.ndarray,       # Shape: [N, 2] 
        r: int, 
        cnt: int, 
        max_iter: int = 10000, 
        seed=42
    ):
        """
        Return: np.ndarray - [cnt, 2] Sample points array
        """
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]
        
        if seed is not None:
            np.random.seed(seed)
        
        given_points = np.array(traj, dtype=float)
        sampled = []
        tries = 0

        while len(sampled) < cnt and tries < max_iter:
            tries += 1
            
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            new_point = np.array([x, y])
            
            dist_to_given = np.linalg.norm(given_points - new_point, axis=1)
            dist_to_sampled = np.linalg.norm(np.array(sampled) - new_point, axis=1) if sampled else np.array([r+1])
            
            if np.all(dist_to_given > r) and np.all(dist_to_sampled > r):
                sampled.append(new_point)

        return np.array(sampled)


    def __call__(
        self,
        points: np.ndarray,     # Shape: [N, 3]
        image: str | np.ndarray,
        distance: int = 100,
        count: int = 4,
    ):
        """
        Given handfree points and mask image, return stacked tracks array including handfree and static points.

        Args:
        points: 
            [N, 3] array of handfree points
        mask_image: 
            path to mask image or numpy array of shape [H, W]
        distance: 
            minimum distance from trajectory points for static points
        count: 
            number of black region points to sample
        """
        # Process handfree points
        handfree_pts = self._pad_points(points)

        # Process static points
        if isinstance(image, str):
            img = Image.open(image).convert('L')

        static_pts_list = self._sample_static_points(
            image=img, traj=points[:, :2], r=distance, cnt=count
        )
        pts_list = [handfree_pts] + [
            self._transfer_to_static_array(pos)
            for pos in static_pts_list
        ]

        return np.stack(pts_list, axis=0) * self.quant_multi
        

    def debug(self, track, image_path, r=30, output_path=None):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for p in track:
            x, y = p[0, :2]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
        if output_path:
            img.save(output_path)
        return img
    

def ati_track(
    model,
    processor,
    image_a_path: str,
    image_b_path: str,
    object: str,
    save_ati_path: str,
    save_pos_dir: str = None,
    delta_t: float = 3 * (1.0 / 30.0),
    frame_num: int = 120,
    map_anything_model: str = "facebook/map-anything"
):
    motion = Motion(
        model=model, processor=processor,
        map_anything_model=map_anything_model
    )
    manager = AtiTrackManager()

    pixels, r = motion.linear_motion(
        image_path_a=image_a_path,
        image_path_b=image_b_path,
        object=object,
        delta_t=delta_t,
        frame_num=frame_num,
        save_pos_dir=save_pos_dir
    )
    pixels = torch.tensor(pixels)
    pixels = torch.cat([pixels, torch.zeros(pixels.size(0), 1)], dim=1).cpu().numpy()
    points = manager(pixels, image_a_path, distance=r, count=4)

    manager.save(points, save_ati_path, compressed=True)
    return points


