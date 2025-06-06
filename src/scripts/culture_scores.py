from PIL import Image
import torch
import pickle
from src.utils import save_pickle
from tqdm import tqdm
from pathlib import Path
from src.config import OUTPUT_PATH, PROMPT_TEMPLATE
from src.models.model_loader import load_model
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def qwen_vl_scores(args):

    model, processor, device = load_model('qwen_vl')

    with open(Path(OUTPUT_PATH) / f"{args.timestamp}" / "WIKI.pkl", 'rb') as f:
        x = pickle.load(f)

    image_paths = sorted(args.image_paths)

    targets = args.target_list

    OUTPUTS = []

    token_ids = [processor.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(image_paths, desc="Processing Images")):
            OP = []

            wiki = x[idx][0]['text'][:15000]  # Truncated for GPU memory
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
                torch.cuda.empty_cache()

            OUTPUTS.append(OP)

    SCORES = []
    for n, a in enumerate(OUTPUTS):
        prompt_scores = {i[0]: i[1].index(max(i[1])) + 1 for i in a}
        SCORES.append({'image_path': image_paths[n], 'values': prompt_scores})
    
    save_pickle(Path(OUTPUT_PATH) / f'{args.timestamp}' / '1-5_scores_VLM_qwen.pkl', SCORES, "1-5 scores")