from PIL import Image
import torch
import pickle
from src.utils import get_image_paths
from tqdm import tqdm
from pathlib import Path
from src.config import OUTPUT_PATH, PROMPT_TEMPLATE
from src.models.model_loader import load_model

def qwen_vl_scores(bp, dataset, target_list):

    model, processor, device = load_model('qwen_vl')

    with open(Path(OUTPUT_PATH) / f'caire_{dataset}_WIKI.pkl', 'rb') as f:
        x = pickle.load(f)

    folder_path = Path(bp) / dataset
    image_paths = sorted(get_image_paths(folder_path))

    with open(target_list, 'rb') as f:
        targets = pickle.load(f)

    OUTPUTS = []

    token_ids = [processor.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(image_paths, desc="Processing Images")):
            OP = []

            wiki = x[idx][0]['text'][:20000]  # Truncated for GPU memory
            entity = x[idx][0]['title']

            image = Image.open(img_path).convert("RGB")

            for target in targets:
                PROMPT = PROMPT_TEMPLATE.format(target=target.capitalize(), wiki=wiki, entity=entity)

                conversation = [
                    {"role": "system", "content": "You are an expert in evaluating the cultural relevance of images."},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
                ]

                text = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)
                token_probs = [probabilities[0, -1, token_id].item() for token_id in token_ids]

                OP.append([target, token_probs])

            OUTPUTS.append(OP)

    with open(Path(OUTPUT_PATH) / f'1-5_{dataset}_VLM_qwen.pkl', 'wb') as f:
        pickle.dump(OUTPUTS, f)
