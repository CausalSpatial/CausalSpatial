import torch
import json
import os
import base64
import argparse
import warnings
import anthropic
import ast
import cv2

from tqdm import tqdm
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText, 
    Qwen2_5_VLForConditionalGeneration, 
    AutoModelForCausalLM
)
from PIL import Image
from qwen_vl_utils import process_vision_info
from openai import OpenAI
from datasets import load_dataset, concatenate_datasets
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

warnings.filterwarnings("ignore")




def pil_to_base64(img, format="PNG"):
    buffer = BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_bytes(img, format="PNG"):
    buffer = BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def get_frames_from_video(video_path, target_frame=[2, 4]):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Cannot open video file!")
        return None, None
    images = []

    for idx in target_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            images.append(pil_img)
        else:
            print(f"Fail to extract {idx+1}th frame.")
    cap.release()
    return images


def load_pretrained(path):
    path_lower = path.lower()
    if "claude" in path_lower:
        return anthropic.Anthropic(), None
    elif "gpt-5" in path_lower or "gpt-4" in path_lower:
        from openai import OpenAI
        return OpenAI(), None
    elif "gemini" in path_lower:
        from google import genai
        return genai.Client(vertexai=True, location='us-central1'), None
    
    print(f"Loading local model: {path}")
    if "llava" in path_lower:
        model_cls = AutoModelForCausalLM
    elif "qwen2.5" in path_lower or "spaceom" in path_lower:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        model_cls = AutoModelForImageTextToText # Qwen old

    try:
        model = model_cls.from_pretrained(
            path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True,
            attn_implementation="flash_attention_2" 
        )
    except:
        model = model_cls.from_pretrained(
            path, 
            torch_dtype=torch.float16, 
            device_map="cuda:0", 
            trust_remote_code=True
        )

    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            
    model.eval()
    return model, processor


def batch_inference(batch_messages, batch_images, path, model, processor, tokens=8192):
    path_lower = path.lower()
    
    # API call models
    is_api = any(x in path_lower for x in ["gpt", "claude", "gemini"])
    
    if is_api:
        def _single_api_call(args):
            index, msg = args
            try:
                if "claude" in path_lower:
                    response = model.messages.create(model=path, max_tokens=tokens, messages=msg)
                    return (index, response.content[0].text)
                
                elif "gpt" in path_lower or "235b" in path_lower:
                    client = model if "235b" not in path_lower else OpenAI(api_key="EMPTY", base_url="http://c008:22002/v1")
                    model_name = path if "235b" not in path_lower else client.models.list().data[0].id
                    response = client.chat.completions.create(model=model_name, messages=msg)
                    return (index, response.choices[0].message.content)
                
                elif "gemini" in path_lower:
                    if isinstance(msg, list) and "content" in msg[0]:
                        for item in msg[0]["content"]:
                            if item.get("type") == "text":
                                prompt_text = item["text"]
                    
                    img = None
                    for item in msg[0]["content"]:
                        if item.get("type") == "image":
                            img = item["image"]
                    image_bytes = pil_to_bytes(img, format="PNG") 
                    contents = [
                        types.Part.from_text(text=prompt_text),
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                    ]
                    config = types.GenerateContentConfig(max_output_tokens=tokens)
                    
                    response = model.models.generate_content(
                        model=path,
                        contents=contents,
                        config=config
                    )
                    return (index, response.text)
                    
            except Exception as e:
                return (index, f"Error: {str(e)}")


        indexed_messages = [(i, msg) for i, msg in enumerate(batch_messages)]
        
        results_map = {}
        with ThreadPoolExecutor(max_workers=len(batch_messages)) as executor:
            futures = executor.map(_single_api_call, indexed_messages)
            
            for idx, result_text in futures:
                results_map[idx] = result_text
        
        sorted_results = [results_map[i] for i in range(len(batch_messages))]
        return sorted_results

    else:
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            if "llava" in path_lower:
                output_texts = [t.split("assistant")[-1].strip() for t in output_texts]
                
            return output_texts
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--subset", nargs="+", type=str, default=["collision"])
    parser.add_argument("--COW", action='store_true')
    parser.add_argument("--COW_output", type=str, default=None)
    parser.add_argument("--video_frame", nargs="+", type=str, default=[1,3,5])
    
    repo_id = "Mwxinnn/CausalSpatial"
    args = parser.parse_args()
    
    model, processor = load_pretrained(args.model_path)
    
    if len(args.subset) == 1:
        dataset = load_dataset(repo_id, args.subset[0], split="train")
    else:
        dataset_list = [
            load_dataset(repo_id, subset, split="train") 
            for subset in args.subset
        ]
        dataset = concatenate_datasets(dataset_list)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    print(f"Starting inference: {len(dataset)} samples, Batch Size: {args.batch_size}")

    # Record extra video frames for inference
    video_frame_dict = {}
    for item in dataset:
        video_frame_dict[item["id"]] = []
    if args.COW:
        video_list = os.listdir(args.COW_output)
        for video_dir in video_list:
            video_path = os.path.join(args.COW_output, video_dir, "gen_video.mp4")
            video_frames = get_frames_from_video(video_path, args.video_frame)
            video_frame_dict[video_dir] = video_frames

    # Evaluate
    collect_results = []
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch_data = dataset[i : i + args.batch_size] 
        
        batch_questions = batch_data["question"]
        batch_images = batch_data["image"]
        batch_ids = batch_data["id"]
        batch_answers = batch_data["answer"]
        batch_not_sure = batch_data["not_sure"]
        
        batch_messages = []
        current_batch_imgs = []
        
        for qid, q, img in zip(batch_ids, batch_questions, batch_images):
            if not args.COW:
                prompt = q + "\n\nWrite your response into this json template: {'Reasoning': '<your reasons>', 'Answer': '<Your answer>'}"
            else:
                prompt = q + "Note that, the first image is the initial state. The subsequent images are simulated scenarios according to the first image and motion context provided in the question. These two images might help you to analyze and reason the question." + "\n\nWrite your response into this json template: {'Reasoning': '<your reasons>', 'Answer': '<Your answer>'}"
            
            use_base64 = any(x in args.model_path.lower() for x in ["gpt", "claude", "qwen3-vl-235b"])
            content = [
                {"type": "image", "image": img}
            ] + [
                {"type": "image", "image": frame} for frame in video_frame_dict[qid]
            ] + [{"type": "text", "text": prompt}]
            
            if use_base64:
                imgs_b64 = [pil_to_base64(img_) for img_ in [img] + [frame for frame in video_frame_dict[qid]]]
                if "claude" in args.model_path.lower():
                    msg = [{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}
                        for img_b64 in imgs_b64
                    ] + [
                        {"type": "text", "text": prompt}
                    ]}]
                elif "gpt" in args.model_path.lower() or "235b" in args.model_path.lower():
                    msg = [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        for img_b64 in imgs_b64
                    ] + [
                        {"type": "text", "text": prompt}
                    ]}]
                else:
                    msg = [{"role": "user", "content": content}]
                batch_messages.append(msg)
                current_batch_imgs.append(img) 
            else:
                msg = [{"role": "user", "content": content}] 
                batch_messages.append(msg)
                current_batch_imgs.append(img)

        try:
            predictions = batch_inference(
                batch_messages, current_batch_imgs, args.model_path, model, processor
            )
            with open(args.output_file, 'a', encoding='utf-8') as f:
                for j, pred in enumerate(predictions):
                    result = {
                        "question_id": batch_ids[j],
                        "question": batch_questions[j], 
                        "gt_answer": batch_answers[j],
                        "model_answer": pred,
                        "model": args.model_path,
                        "not_sure": batch_not_sure[j]
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    collect_results.append(result)
                    
        except Exception as e:
            print(f"Batch Error at index {i}: {e}")
            continue

    # Score
    print("=" * 50)
    print(f"Model: {args.model_path}")

    tmp = {
    "L1": {"collision": 0, "physics": 0, "occlusion": 0, "compatibility": 0},
    "L2": {"collision": 0, "physics": 0, "occlusion": 0, "compatibility": 0}
    }
    res = tmp.copy()
    num = tmp.copy()
    error = tmp.copy()

    for item in collect_results:
        subset = item["type"]
        level = item["difficulty"]
        
        num[level][subset] += 1
        ans = item["gt_answer"][1] if item["gt_answer"][0] == "(" else item["gt_answer"][0]

        try:
            start_idx = item["model_answer"].find('{')
            end_idx = item["model_answer"].rfind('}')
            output = ast.literal_eval(item["model_answer"][start_idx: end_idx+1])
            answer = output["Answer"][0]
        except:
            error[level][subset] += 1

        if ans == answer:
            res[level][subset] += 1

    for level, level_res in res.items():
        for subset, score in level_res.items():
            s = score/num[level][subset] if num[level][subset] != 0 else "NaN"
            print(f"{level} {subset}:\t{s:.2%} ({score} / {num[level][subset]})")
    
    print("=" * 50)

if __name__ == "__main__":
    main()