import json
import re
import copy

import numpy as np

from PIL import Image
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from matplotlib import pyplot as plt
import matplotlib.patches as patches




class BboxExtractor:
    def __init__(
        self,
        model=None,
        processor=None
    ):
        self.model = model
        self.processor = processor

        self.system_prompt = {
            "role": "system",
            "content": [{
                "type": "text", 
                "text": (
                    "You are a grounding assistant. Return ONLY valid JSON with the schema."
                    "All coordinates are absolute pixel values on the processed image. Do not output text.")}]}

        self.prompt_identify = """# Role
You are an advanced visual perception assistant specializing in analyzing physical dynamics and object identification from images and text.

# Task
Identify the target object of interest based on the User Input Text and the provided Image.

# Identification Rules
1. **Focus on Motion:** Identify the object that is explicitly described as moving, about to move, or is the subject of a motion query (e.g., "If X moves...").
2. **Resolve Ambiguity via Image:**
   - The text description might be vague (e.g., "the ball").
   - Use visual indicators in the image (Bounding Boxes, Arrows, Motion blur, or Highlights) to pinpoint the exact object.
   - **Conflict Resolution:** If the text says "the ball" but the image shows a specific "blue striped 10 ball" at the referenced location, YOU MUST USE THE DETAILED VISUAL DESCRIPTION.
3. **Billiard Specifics:** For billiard balls, you must specify: Color + Number + Type (Solid/Striped).
4. **Trajectory & Multi-Instance Sequences:**
   - If the image displays a **trajectory** (multiple instances of the same object connected by lines or arrows showing a path):
   - You MUST identify the object at the **STARTing point** of the trajectory.
   - **Modifier Requirement:** You MUST append a specific spatial modifier to distinguish this instance from others in the trajectory (e.g., "the basketball furthest from the hoop", "the ball on the far left of the yellow line"). Do NOT just output "the basketball".
5. Note that, DO NOT ANSWER THE GIVEN QUESTION!!! If the input text is a question, you should analyze the moving object implied in the question and image, instead of reasoning the question to get answer.

# Output Format
You must structure your response in two parts:
1. **Reasoning:** Briefly explain which object you identified and why (referencing bbox or visual features).
2. **Final Result:** A JSON format, list of strings containing ONLY the identified object name(s).

# Examples

User Input: "The blue ball is rolling towards the red cube. What should be removed?"
Assistant Response:
Reasoning: The text asks what should be removed (likely the cube), but the object performing the action "rolling" is the blue ball.
Final Result: \n```json\n["the blue ball"]```

User Input: "Can the wooden calendar be removed if the white car moves backward?"
Assistant Response:
Reasoning: The text mentions the "white car" moving. In the image, the white car is located near the calendar. The moving object is the car.
Final Result: \n```json\n["the white car"]```

User Input: "If the toy car moves forward, which object should be removed to avoid the collision?"
Assistant Response:
Reasoning: The text mentions the "toy car" moving. In the image, the white car is located near the vase. The moving object is the car.
Final Result: \n```json\n["the toy car"]```

User Input: "If the red sphere rolls to the right, will it hit the wall?"
Assistant Response:
Reasoning: The text specifies a "red sphere" rolling.
Final Result: \n```json\n["the red sphere"]```

User Input: "What is the ball inside the bbox?" (Image shows a blue striped ball with number 10)
Assistant Response:
Reasoning: The user refers to a bbox. Visually, inside the bbox is a blue ball with a stripe and the number 10.
Final Result: \n```json\n["blue striped 10 ball"]```

---
# User Input
{}"""

        self.prompt_locate = "Please locate the {} in the image with its bbox coordinates and its label and output in JSON format."

        self.prompt_rewrite = "Rewrite the given question into a concise declarative sentence that directly describes the motion mentioned in the question.\n" \
            "Do not add any extra information or commentary." \
            "Return the result in a Python list format as shown below." \
            "Example:\n" \
            "Input: \"If the apple falls freely from its current position, can it fit into the wooden bowel? 'Fit into' means it doesn't exceed the horizontal plane of the wooden bowel's opening. Answer by (A) Yes, (B) No or (C) Not sure.\"" \
            "Output: [\"The apple falls freely from its current position.\"]" \
            "---\n" \
            "Please Answer:\n" \
            "Input:\n{}"


    def _to_pixel(self, v, axis_len: int) -> int:
        try:
            fv = float(v)
        except:
            return 0
        if 0.0 <= fv <= 1.0:
            px = fv * axis_len
        elif 0.0 <= fv <= 1000.0:
            px = fv / 1000.0 * axis_len
        else:
            px = fv
        return int(max(0, min(axis_len - 1, round(px))))


    def _strip_code_fence(self, text: str) -> str:
        if "```" not in text:
            return text
        m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.S)
        return m.group(1).strip() if m else text


    def _try_parse_json_points(self, text: str):
        t = self._strip_code_fence(text)
        try:
            data = json.loads(t)
            if not isinstance(data, list):
                data = [data]
            pts, labs = [], []
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                    
                key = None
                for k in ["bbox_2d", "point_2d", "point2d", "point", "coords", "coordinate", "coord"]:
                    if k in item:
                        key = k
                        break
                if key is None:
                    continue
                arr = item[key]
                if (isinstance(arr, (list, tuple)) and len(arr) >= 2
                    and all(isinstance(v, (int, float)) for v in arr[:2])):
                    pts.append(arr)
                    labs.append(item.get("label", f"point_{len(pts)}"))
            return pts, labs
        except Exception:
            return [], []


    def _regex_fallback_points(self, text: str):
        labels = []
        for m in re.finditer(r'(?i)(?:label|name)\s*[:：]\s*["“]?([^\s"”]+)["”]?', text):
            labels.append((m.start(), m.group(1)))

        points = []
        point_spans = []

        # 1) [x, y] (x, y)
        for m in re.finditer(r'[\[\(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[\]\)]', text):
            points.append([float(m.group(1)), float(m.group(2))])
            point_spans.append(m.span())

        # 2) x: ..., y: ...
        for m in re.finditer(r'(?i)x\s*[:：]\s*(-?\d+(?:\.\d+)?)\s*[,，;\s]+y\s*[:：]\s*(-?\d+(?:\.\d+)?)', text):
            points.append([float(m.group(1)), float(m.group(2))])
            point_spans.append(m.span())

        assigned_labels = []
        for i, span in enumerate(point_spans):
            nearest = None
            nearest_dist = None
            for (pos, lab) in labels:
                d = min(abs(pos - span[0]), abs(pos - span[1]))
                if nearest is None or d < nearest_dist:
                    nearest = lab
                    nearest_dist = d
            assigned_labels.append(nearest if nearest else f"point_{i+1}")

        return points, assigned_labels


    def extract_points_for_plot(
        self,
        qwen_answer: str,
        im_or_size
    ):
        if hasattr(im_or_size, "size"):
            width, height = im_or_size.size
        else:
            width, height = im_or_size  # (w, h)

        pts, labs = self._try_parse_json_points(qwen_answer)
        if not pts:
            pts, labs = self._regex_fallback_points(qwen_answer)
        if not pts:
            return []

        if len(labs) < len(pts):
            labs += [f"point_{i+1}" for i in range(len(labs), len(pts))]

        results = []
        for (i, (x1, y1, x2, y2)) in enumerate(pts):
            results.append({
                "x1": x1 / 1000 * width, 
                "y1": y1 / 1000 * height,
                "x2": x2 / 1000 * width, 
                "y2": y2 / 1000 * height, 
                "label": labs[i]
            })
        return results


    def draw_bbox(self, image_path, point_dict, output_path, color='red', linewidth=2):
        img = np.array(Image.open(image_path))

        points = [
            [point_dict["x1"], point_dict["y1"]],
            [point_dict["x2"], point_dict["y1"]],
            [point_dict["x2"], point_dict["y2"]],
            [point_dict["x1"], point_dict["y2"]],
        ]

        points = np.array(points)
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()
        width, height = x_max - x_min, y_max - y_min

        fig, ax = plt.subplots()
        ax.imshow(img)
        
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=linewidth, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.scatter(points[:, 0], points[:, 1], c=color, s=30)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close(fig)


    def generate(self, messages):
        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


    def get_object(self, prompt, image=None):
        messages = [{
            "role": "user",
            "content": (
                [{"type": "image", "image": image}] if image else []
            ) + [{"type": "text", "text": self.prompt_identify.format(prompt)}]
        }]
        output_text = self.generate(messages)[0]
        output = self._strip_code_fence(output_text)
        return json.loads(output)
    

    def get_detailed_prompt(self, prompt):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": self.prompt_rewrite.format(prompt)}]
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        output = self._strip_code_fence(output_text)
        return json.loads(output)


    def __call__(
        self,
        image_path: str,
        prompt: str = None,
        object: str = None,
    ):
        messages = [self.system_prompt] + [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": self.prompt_locate.format(object)}
        ]}]
        
        answer = self.generate(messages)[0]
        img = Image.open(image_path)
        points = self.extract_points_for_plot(answer, img)
        return points


