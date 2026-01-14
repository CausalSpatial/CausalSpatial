import torch
import argparse
import os
import pprint

from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from asset.trajectory import Motion
from asset.get_ati_track import AtiTrackManager
from asset.ati_pipeline import ATI




class ObjWM:
    def __init__(
        self,
        frame_num: int = 60,
        delta_t: float = 3 * (1.0 / 30.0),
        model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        map_anything_model: str = "facebook/map-anything",
        debug: bool = True,
        mllm_device: str = "cuda:0",
        video_device: str = "cuda:1"
    ):
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model, dtype="auto", device_map=mllm_device)
        self.processor = AutoProcessor.from_pretrained(model)
        self.motion = Motion(
            model=self.model, processor=self.processor,
            map_anything_model=map_anything_model
        )
        self.manager = AtiTrackManager()
        
        self.ati = ATI().to(video_device)

        self.frame_num = frame_num
        self.delta_t = delta_t
        self.debug = debug
        

    def __call__(
        self,
        prompt,
        save_dir: str,
        image_a_path: str,
        image_b_path: str = None,
        motion_type: str = "linear",    # question category, ["linear", "parabolic"]
        input_type: str = "single",     # input image number, ["single", "dual"]
        velocity_abs: int = 500,
        generate: bool = False
    ):
        obj = self.motion.bbox_extractor.get_object(prompt, image_a_path)
        detailed_prompt = self.motion.bbox_extractor.get_detailed_prompt(prompt)

        dual_params = {
            "image_path_a": image_a_path,
            "image_path_b": image_b_path,
            "object": obj[0],
            "delta_t": self.delta_t,
            "frame_num": self.frame_num,
            "save_pos_dir": save_dir,
            "debug": self.debug
        }

        single_params = {
            "image_path": image_a_path,
            "object": obj[0],
            "motion_context": detailed_prompt,
            "velocity_abs": velocity_abs,
            "frame_num": self.frame_num,
            "save_pos_dir": save_dir,
            "debug": self.debug
        }

        if motion_type == "linear" and input_type == "dual":
            pixels, r = self.motion.linear_motion(**dual_params)
        elif motion_type == "linear" and input_type == "single":
            pixels, r = self.motion.linear_motion_single(**single_params)
        elif motion_type == "parabolic" and input_type == "dual":
            pass
        elif motion_type == "parabolic" and input_type == "single":
            pixels, r = self.motion.parabolic_motion_single(**single_params)

        pixels = torch.tensor(pixels)
        pixels = torch.cat([pixels, torch.zeros(pixels.size(0), 1)], dim=1).cpu().numpy()
        points = self.manager(pixels, image_a_path, distance=r, count=4)

        save_ati_path = os.path.join(save_dir, "output.pth")        
        self.manager.save(points, save_ati_path, compressed=True)

        if generate:
            self.ati(
                image=image_a_path,
                prompt=detailed_prompt,
                trajectory=save_ati_path,
                save_path=os.path.join(save_dir, "gen_video.mp4")
            )

        return {
            "save": save_ati_path,
            "object": obj,
            "rewrite_prompt": detailed_prompt
        }
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subset", nargs="+", type=str, default=["collision"])
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    mllm_gpu_id = local_rank * 2
    video_gpu_id = local_rank * 2 + 1

    mllm_device = f"cuda:{mllm_gpu_id}"
    video_device = f"cuda:{video_gpu_id}"

    repo_id = "Mwxinnn/CausalSpatial"
    args = parser.parse_args()
    
    if len(args.subset) == 1:
        dataset = load_dataset(repo_id, args.subset[0], split="train")
    else:
        dataset_list = [
            load_dataset(repo_id, subset, split="train") 
            for subset in args.subset
        ]
        dataset = concatenate_datasets(dataset_list)
    
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=global_rank)
        print(f"Rank {global_rank}: Processing {len(dataset)} samples...")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print("Output directory already exists. Exiting...")
        exit(0)
    
    iwm = ObjWM(
        mllm_device=mllm_device,
        video_device=video_device
    )

    for item in tqdm(dataset, desc=f"Rank {global_rank}", position=global_rank):
        sample_save_dir = os.path.join(args.output_dir, str(item["id"]))
        if not os.path.exists(sample_save_dir):
             os.makedirs(sample_save_dir, exist_ok=True)

        res = iwm(
            prompt=item["question"],
            image_a_path=item["image"],
            save_dir=sample_save_dir,
            generate=True
        )

        if global_rank == 0:
            pprint.pprint(res, indent=4, sort_dicts=False)
            print("=" * 50)




if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     import json
#     import os

#     data_idx = 0
#     # data_idx = 7

#     data_path = "/scratch/dwirtz1/wcloong-iwm/Benchmark_inference/collision"
#     with open(os.path.join(data_path, "level_1.jsonl"), "r") as f:
#         data = [json.loads(l) for l in f]
    
#     data = data[data_idx]
#     question = data["question"]
#     image_path = os.path.join(data_path, data["image"])


#     iwm = ObjWM()

#     res = iwm(
#         prompt=question,
#         image_a_path=image_path,
#         save_dir="/scratch/dwirtz1/wcloong-iwm/test-result",
#         # motion_type="parabolic"
#     )
