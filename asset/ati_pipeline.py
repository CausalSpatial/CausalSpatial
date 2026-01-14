import os
import sys
import logging
import random
import warnings
from datetime import datetime
from typing import Optional

import torch
import torch.distributed as dist
from PIL import Image

import sub_module.ati.wan as wan
from sub_module.ati.wan.configs import (
    MAX_AREA_CONFIGS, 
    SIZE_CONFIGS, 
    SUPPORTED_SIZES, 
    WAN_CONFIGS
)
from sub_module.ati.wan.utils.motion import get_tracks_inference
from sub_module.ati.wan.utils.utils import cache_video, str2bool
from sub_module.ati.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

warnings.filterwarnings("ignore")


class ATI:
    def __init__(
        self,
        task: str = "ati-14B",
        size: str = "832*480",
        ckpt_dir: Optional[str] = "../Wan2.1-ATI-14B-480P",
        *,
        offload_model: Optional[bool] = None,
        ulysses_size: int = 1,
        ring_size: int = 1,
        t5_fsdp: bool = False,
        t5_cpu: bool = False,
        dit_fsdp: bool = False,
        frame_num: Optional[int] = None,
        sample_solver: str = "unipc",
        sample_steps: Optional[int] = None,
        sample_shift: Optional[float] = None,
        sample_guide_scale: float = 5.0,
        base_seed: int = -1,
        use_prompt_extend: bool = False,
        prompt_extend_method: str = "local_qwen",  # ["dashscope", "local_qwen"]
        prompt_extend_model: Optional[str] = None,
        prompt_extend_target_lang: str = "zh",
    ) -> None:
        self.task = task
        self.size = size
        self.ckpt_dir = ckpt_dir
        self.offload_model = offload_model
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        self.t5_fsdp = t5_fsdp
        self.t5_cpu = t5_cpu
        self.dit_fsdp = dit_fsdp
        self.frame_num = frame_num
        self.sample_solver = sample_solver
        self.sample_steps = sample_steps
        self.sample_shift = sample_shift
        self.sample_guide_scale = sample_guide_scale
        self.base_seed = base_seed
        self.use_prompt_extend = use_prompt_extend
        self.prompt_extend_method = prompt_extend_method
        self.prompt_extend_model = prompt_extend_model
        self.prompt_extend_target_lang = prompt_extend_target_lang

        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device_id = self.local_rank

        self._init_logging()
        self._validate_and_defaults()
        self._init_distributed_and_parallel()
        self._init_prompt_expander()
        self._init_pipeline()


    def _init_logging(self):
        if self.rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s: %(message)s",
                handlers=[logging.StreamHandler(stream=sys.stdout)],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

    def _validate_and_defaults(self):
        if self.sample_steps is None:
            self.sample_steps = 40
        if self.sample_shift is None:
            self.sample_shift = 5.0
        if self.frame_num is None:
            self.frame_num = 1 if "t2i" in self.task else 81

        if "t2i" in self.task:
            assert self.frame_num == 1, f"task={self.task} Only support frame_num=1"

        if self.base_seed is None or self.base_seed < 0:
            self.base_seed = random.randint(0, sys.maxsize)

        assert (
            self.size in SUPPORTED_SIZES[self.task]
        ), f"Unsupported size={self.size}\n task={self.task}；Options: {', '.join(SUPPORTED_SIZES[self.task])}"

    def _init_distributed_and_parallel(self):
        if self.offload_model is None:
            self.offload_model = False if self.world_size > 1 else True
            logging.info(f"[Init] offload_model Undefine, Default {self.offload_model}")

        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://", rank=self.rank, world_size=self.world_size)
        else:
            assert not (self.t5_fsdp or self.dit_fsdp), "Not supported in non-distributed environments: t5_fsdp / dit_fsdp"
            assert not (self.ulysses_size > 1 or self.ring_size > 1), "Not supported in non-distributed environments: ulysses/ring"

        if self.ulysses_size > 1 or self.ring_size > 1:
            assert self.ulysses_size * self.ring_size == self.world_size, "ulysses_size * ring_size should be equal to world_size"
            from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

            init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=self.ring_size,
                ulysses_degree=self.ulysses_size,
            )

        if dist.is_initialized():
            base_seed = [self.base_seed] if self.rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            self.base_seed = base_seed[0]


    def _init_prompt_expander(self):
        self.prompt_expander = None
        if not self.use_prompt_extend:
            return
        if self.prompt_extend_method == "dashscope":
            self.prompt_expander = DashScopePromptExpander(
                model_name=self.prompt_extend_model, is_vl=True
            )
        elif self.prompt_extend_method == "local_qwen":
            self.prompt_expander = QwenPromptExpander(
                model_name=self.prompt_extend_model, is_vl=True, device=self.rank
            )
        else:
            raise NotImplementedError(f"Unsupported prompt_extend_method: {self.prompt_extend_method}")

    def _init_pipeline(self):
        cfg = WAN_CONFIGS[self.task]
        if self.ulysses_size > 1:
            assert cfg.num_heads % self.ulysses_size == 0, f"`cfg.num_heads` must be divisible by ulysses_size."

        logging.info(f"[Init] Model Configuration: {cfg}")

        self.cfg = cfg
        self.pipeline = wan.WanATI(
            config=cfg,
            checkpoint_dir=self.ckpt_dir,
            device_id=self.device_id,
            rank=self.rank,
            t5_fsdp=self.t5_fsdp,
            dit_fsdp=self.dit_fsdp,
            use_usp=(self.ulysses_size > 1 or self.ring_size > 1),
            t5_cpu=self.t5_cpu,
        )


    def _maybe_extend_prompt(self, prompt: str, img: Optional[Image.Image]) -> str:
        if not self.use_prompt_extend or self.prompt_expander is None:
            return prompt

        logging.info("[Prompt] Starting expanding ...")
        input_prompt = [prompt]  # fallback
        if self.rank == 0:
            result = self.prompt_expander(
                prompt,
                tar_lang=self.prompt_extend_target_lang,
                image=img,
                seed=self.base_seed,
            )
            if getattr(result, "status", False):
                input_prompt = [result.prompt]
            else:
                logging.info(f"[Prompt] Expansion failure：{getattr(result, 'message', 'unknown')}, back to initial prompt。")
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)

        logging.info(f"[Prompt] Expanded prompt: {input_prompt[0]}")
        return input_prompt[0]


    def __call__(
        self, 
        image: str,     # path of image, representing the initial state
        prompt: str,    # 
        trajectory: Optional[str], 
        save_path: str  
    ) -> str:
        if os.path.exists(save_path):
            logging.info(f"[Save] {save_path} exists. Skip Generation")
            return save_path

        logging.info(f"[Input] prompt: {prompt}")
        logging.info(f"[Input] image:  {image}")
        img = Image.open(image).convert("RGB")
        width, height = img.size
        tracks = get_tracks_inference(trajectory, height, width)

        prompt = self._maybe_extend_prompt(prompt, img)

        logging.info("[Gen] Start Generation ...")
        video = self.pipeline.generate(
            prompt,
            img,
            tracks,
            max_area=MAX_AREA_CONFIGS[self.size],
            frame_num=self.frame_num,
            shift=self.sample_shift,
            sample_solver=self.sample_solver,
            sampling_steps=self.sample_steps,
            guide_scale=self.sample_guide_scale,
            seed=self.base_seed,
            offload_model=self.offload_model,
        )

        if self.rank == 0:
            root, ext = os.path.splitext(save_path)
            if not ext:
                save_path = root + ".mp4"

            os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
            logging.info(f"[Save] Results save to: {save_path}")
            cache_video(
                tensor=video[None],
                save_file=save_path,
                fps=self.cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        logging.info("[Gen] Finish Generation")
        return save_path


