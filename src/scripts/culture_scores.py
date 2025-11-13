from PIL import Image
import torch
import pickle
import json
import re
import logging
from src.utils import save_pickle
from tqdm import tqdm
from pathlib import Path
from src.config import OUTPUT_PATH, PROMPT_TEMPLATE, PROMPT_TEMPLATE_MULTI, USE_MULTIPLE_WIKI_PAGES, WIKI_CHARS_PER_PAGE
from src.models.model_loader import load_model
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_json_response(response_text):
    """Parse JSON response from VLM, with fallback for malformed responses"""
    try:
        # Clean control characters that can break JSON parsing
        cleaned_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', response_text)
        
        # Try to find JSON in the response
        json_match = re.search(r'\{[^}]*"score"[^}]*"reasoning"[^}]*\}', cleaned_text)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                score = data.get('score')
                if score is not None:
                    score = int(score)
                    # Ensure score is between 1-5
                    score = max(1, min(5, score))
                else:
                    score = None  # Return None instead of default 3
                reasoning = data.get('reasoning', 'No reasoning provided')
                return score, reasoning
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"JSON parsing failed: {str(e)}, attempting fallback")
        
        # Try to extract score from text as fallback
        score_match = re.search(r'["\']?score["\']?\s*:\s*(\d+)', cleaned_text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
            score = max(1, min(5, score))
            return score, cleaned_text[:200]  # Use first 200 chars as reasoning
        else:
            # Last resort: return None (N/A) instead of 3
            return None, f"Could not parse response: {cleaned_text[:200]}"
    except Exception as e:
        logging.error(f"Error parsing response: {str(e)}")
        return None, f"Error parsing response: {str(e)}"

def qwen_vl_scores(args, use_multiple_wiki_pages=None, model_name='qwen_vl'):
    """
    Score images for cultural relevance using Vision-Language models.
    
    Args:
        args: Arguments object containing image_paths, target_list, and timestamp
        use_multiple_wiki_pages: Whether to use multiple Wikipedia pages. 
                                 If None, uses the config value USE_MULTIPLE_WIKI_PAGES
        model_name: Name of the model to use ('qwen_vl', 'pangea_vl', 'llama_vl')
                    Defaults to 'qwen_vl' for backwards compatibility
    """
    # Use parameter if provided, otherwise fall back to config
    if use_multiple_wiki_pages is None:
        use_multiple_wiki_pages = USE_MULTIPLE_WIKI_PAGES

    model, processor, device = load_model(model_name)

    with open(Path(OUTPUT_PATH) / f"{args.timestamp}" / "WIKI.pkl", 'rb') as f:
        x = pickle.load(f)

    image_paths = sorted(args.image_paths)

    targets = args.target_list

    OUTPUTS = []

    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(image_paths, desc="Processing Images")):
            OP = []
            
            # Check if Wikipedia pages were retrieved for this image
            if not x[idx] or len(x[idx]) == 0:
                logging.warning(f"No Wikipedia pages found for image {img_path}, skipping scoring")
                # Add default scores for this image
                for target in targets:
                    OP.append([target, 3, "No Wikipedia pages available for this image"])
                OUTPUTS.append(OP)
                continue

            # Prepare Wikipedia context based on mode
            if use_multiple_wiki_pages and len(x[idx]) > 1:
                # Multi-page mode: use up to 10 pages with WIKI_CHARS_PER_PAGE chars each
                entities = []
                wiki_texts = []
                for page_idx, page in enumerate(x[idx][:10]):  # Use up to 10 pages
                    if page and 'title' in page and 'text' in page:
                        entities.append(page['title'])
                        wiki_texts.append(f"**{page['title']}**: {page['text'][:WIKI_CHARS_PER_PAGE]}")
                
                entities_str = ", ".join(entities)
                wiki_combined = "\n\n".join(wiki_texts)
                use_multi_prompt = True
            else:
                # Single-page mode: use only the first page with 15000 chars
                wiki = x[idx][0]['text'][:15000]
                entity = x[idx][0]['title']
                use_multi_prompt = False

            image = Image.open(img_path).convert("RGB")

            for target in targets:
                # Choose prompt template based on mode
                if use_multi_prompt:
                    PROMPT = PROMPT_TEMPLATE_MULTI.format(
                        target=target.capitalize(), 
                        wiki=wiki_combined, 
                        entities=entities_str
                    )
                else:
                    PROMPT = PROMPT_TEMPLATE.format(
                        target=target.capitalize(), 
                        wiki=wiki, 
                        entity=entity
                    )

                # Handle different model types
                if model_name == "pangea_vl":
                    # Pangea model uses manual prompt formatting (no chat template support)
                    text_input = f"<|im_start|>system\nYou are an expert in evaluating the cultural relevance of images.<|im_end|>\n<|im_start|>user\n<image>\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
                    inputs = processor(images=image, text=text_input, return_tensors='pt').to(device)
                else:
                    # Qwen-VL and Llama Vision use chat templates
                    conversation = [
                        {"role": "system", "content": "You are an expert in evaluating the cultural relevance of images."},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
                    ]
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

                # Generate text response instead of using token probabilities
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.7,
                )
                
                # Decode the generated text
                generated_text = processor.batch_decode(
                    generated_ids[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Parse JSON response to extract score and reasoning
                score, reasoning = parse_json_response(generated_text)
                
                OP.append([target, score, reasoning])
                torch.cuda.empty_cache()

            OUTPUTS.append(OP)

    SCORES = []
    for n, a in enumerate(OUTPUTS):
        # Store scores with reasoning
        prompt_scores = {i[0]: i[1] for i in a}  # {culture: score}
        prompt_reasoning = {i[0]: i[2] for i in a}  # {culture: reasoning}
        SCORES.append({
            'image_path': image_paths[n], 
            'values': prompt_scores,
            'reasoning': prompt_reasoning
        })
    
    # Use model_name in output filename
    output_filename = f'1-5_scores_VLM_{model_name}.pkl'
    save_pickle(Path(OUTPUT_PATH) / f'{args.timestamp}' / output_filename, SCORES, "1-5 scores")