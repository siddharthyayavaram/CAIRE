import os
import logging
import tempfile
import shutil
import pickle
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import original CAIRE functions - exactly like main.py
from src.scripts.retrieval import process_images
from src.scripts.disambiguation import lemma_match
from src.scripts.fetch_wikipedia import wiki_retrieval
from src.scripts.culture_scores import qwen_vl_scores
from src.utils import load_model
from src.config import MAX_WIKI_DOCS, OUTPUT_PATH

logger = logging.getLogger(__name__)

class CAIREPipeline:
    def __init__(self):
        self.model = None
        self.processor = None
        self.cache = {}  # Cache for storing intermediate results by session_id
        
    def initialize(self):
        """Initialize models - like main.py line 22-23"""
        logger.info("Initializing model...")
        self.model, self.processor = load_model()
        logger.info("Model initialized successfully")
    
    def process_image(self, image: Image.Image, cultures: List[str], use_multiple_wiki_pages: bool = False, model_name: str = 'qwen_vl', session_id: str = None) -> Dict[str, Any]:
        """Run the CAIRE pipeline - exactly like run_pipeline() in main.py
        
        Args:
            image: Input image to process
            cultures: List of cultures to evaluate
            use_multiple_wiki_pages: Whether to use multiple Wikipedia pages for context
            model_name: VLM model to use for scoring ('qwen_vl', 'pangea_vl', 'llama_vl')
            session_id: Optional session ID to reuse cached intermediate results
        """
        
        # Check if we can reuse cached results
        logger.info(f"Cache check: session_id={session_id}, cache_keys={list(self.cache.keys())}")
        if session_id and session_id in self.cache:
            cached_data = self.cache[session_id]
            logger.info(f"Found cached session {session_id}, analyzing cultures: {cultures}")
            # Image matches (same session) - only cultures, wiki mode, and model can differ
            # All of those only affect scoring, not retrieval/lemma/wikipedia steps
            logger.info(f"✅ Reusing cached data from session {session_id}, only running scoring for cultures={cultures} with {model_name} and wiki_mode={use_multiple_wiki_pages}")
            return self._process_with_cache(cached_data, cultures, use_multiple_wiki_pages, model_name, session_id)
        else:
            logger.info(f"No cache found for session {session_id}, running full pipeline")
        
        # Create args object mimicking command-line args (like main.py lines 42-50)
        class Args:
            pass
        
        args = Args()
        args.target_list = cultures
        args.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        args.is_predefined_list = False
        args.is_folder = False
        
        # Create output directory (like main.py lines 47-48)
        output_dir = Path(OUTPUT_PATH) / args.timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image in output directory for caching
        cached_image_path = output_dir / "cached_image.jpg"
        image.save(cached_image_path)
        args.image_paths = [cached_image_path]
        
        try:
            
            # Run pipeline - EXACT sequence from main.py lines 25-35
            logger.info("Processing images...")
            process_images(args, self.model, self.processor)
            
            logger.info("Performing lemma matching...")
            lemma_match(args)
            
            logger.info("Fetching Wikipedia data...")
            wiki_retrieval(args, MAX_WIKI_DOCS)
            
            logger.info(f"1-5 Scoring with {model_name}...")
            qwen_vl_scores(args, use_multiple_wiki_pages, model_name)
            
            # Read results
            logger.info("Reading results...")
            results = self._read_results(args, model_name)
            
            # Generate session_id and cache intermediate results for reuse
            session_id = args.timestamp
            
            # Cache the output directory and image info for potential reuse
            # Note: We don't cache cultures or wiki_mode since they only affect scoring
            self.cache[session_id] = {
                'output_dir': str(output_dir),
                'timestamp': args.timestamp,
                'image_path': str(cached_image_path),
                'image_hash': hash(image.tobytes())  # To verify same image
            }
            
            # Add session_id to results
            results['session_id'] = session_id
            
            return results
            
        except Exception as e:
            # Clean up on error
            try:
                if output_dir.exists():
                    shutil.rmtree(output_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up on error: {cleanup_error}")
            raise e
    
    def _process_with_cache(self, cached_data: Dict[str, Any], cultures: List[str], use_multiple_wiki_pages: bool, model_name: str, session_id: str) -> Dict[str, Any]:
        """Process only the scoring step using cached intermediate results"""
        
        # Reconstruct args object from cached data
        class Args:
            pass
        
        args = Args()
        args.timestamp = cached_data['timestamp']
        args.target_list = cultures
        args.image_paths = [Path(cached_data['image_path'])]  # Reuse cached image
        
        # Always run scoring since cultures, model, or wiki_mode may have changed
        output_dir = Path(cached_data['output_dir'])
        logger.info(f"Running scoring with {model_name} for cultures={cultures} with wiki_mode={use_multiple_wiki_pages}")
        qwen_vl_scores(args, use_multiple_wiki_pages, model_name)
        
        # Read and return results
        results = self._read_results(args, model_name)
        results['session_id'] = session_id
        
        return results
    
    def _read_results(self, args, model_name: str = 'qwen_vl') -> Dict[str, Any]:
        """Read results from pickle files created by CAIRE pipeline"""
        
        output_dir = Path(OUTPUT_PATH) / args.timestamp
        
        # Read Wikipedia data
        wiki_path = output_dir / "WIKI.pkl"
        with open(wiki_path, 'rb') as f:
            wiki_data = pickle.load(f)
        
        # Read lemma match data (contains scores for each bid)
        lemma_path = output_dir / "lemma_match.pkl"
        with open(lemma_path, 'rb') as f:
            lemma_data = pickle.load(f)
        
        # Read scores - use dynamic filename based on model_name
        scores_path = output_dir / f"1-5_scores_VLM_{model_name}.pkl"
        with open(scores_path, 'rb') as f:
            scores_data = pickle.load(f)
        
        # Format results for API response
        scores_result = []
        reasoning_data = scores_data[0].get('reasoning', {})
        for culture, score in scores_data[0]['values'].items():
            # Handle None scores from models like pangea_vl
            if score is None:
                logger.warning(f"Score for culture '{culture}' is None, skipping")
                continue
            scores_result.append({
                "culture": culture,
                "score": float(score),
                "reasoning": reasoning_data.get(culture, "No reasoning provided")
            })
        
        # Format Wikipedia pages with matching scores from lemma_match
        wiki_pages = []
        for idx, page in enumerate(wiki_data[0][:10]):
            if page and 'title' in page:
                # Get the corresponding lemma match score (if available)
                match_score = lemma_data[0][idx]['score'] if idx < len(lemma_data[0]) else 0.0
                
                wiki_pages.append({
                    "title": page['title'],
                    "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
                    "rank": idx + 1,
                    "score": float(match_score)
                })
        
        # Get matched entity
        matched_entity = wiki_data[0][0]['title'] if wiki_data[0] and wiki_data[0][0] else "Unknown"
        
        return {
            "scores": scores_result,
            "wikipedia_pages": wiki_pages,
            "matched_entity": matched_entity
        }

# Global pipeline instance
pipeline = CAIREPipeline() 