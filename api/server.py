import logging
import os
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from pathlib import Path
import pickle

# Set cache directories relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")

# Set HuggingFace token for accessing gated models (e.g., Llama)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
if not os.environ["HF_TOKEN"]:
    raise ValueError("HF_TOKEN not found in environment variables. Please make sure it's set in your bashrc and the environment is sourced.")

from api.models import (
    AnalysisResponse,
    HealthResponse,
    PredefinedListsResponse,
    CultureScore,
    WikipediaPage
)
from api.api_pipeline import pipeline
from src.config import DATA_PATH, PREDEFINED_TARGET_LISTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CAIRE API",
    description="Cultural Attribution of Images by Retrieval-Augmented Evaluation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Initialize pipeline on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting CAIRE API server...")
    try:
        pipeline.initialize()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        raise

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="CAIRE API is running"
    )

@app.get("/api/predefined-lists", response_model=PredefinedListsResponse)
async def get_predefined_lists():
    """Get list of available predefined culture lists"""
    try:
        lists = [path.name for path in PREDEFINED_TARGET_LISTS]
        return PredefinedListsResponse(lists=lists)
    except Exception as e:
        logger.error(f"Error fetching predefined lists: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch predefined lists")

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    cultures: str = Form(...),
    use_multiple_wiki_pages: bool = Form(False),
    model_name: str = Form("qwen_vl"),
    session_id: Optional[str] = Form(None)
):
    """
    Analyze cultural relevance of an image
    
    Args:
        image: Image file to analyze
        cultures: Comma-separated list of cultures (e.g., "India,China,USA")
        use_multiple_wiki_pages: Whether to use multiple Wikipedia pages in context (default: False)
        model_name: VLM model to use for scoring (default: "qwen_vl")
                    Options: "qwen_vl", "pangea_vl", "llama_vl"
        session_id: Optional session ID to reuse cached intermediate results (for model switching)
    
    Returns:
        Analysis results with cultural scores and Wikipedia pages
    """
    logger.info(f"Received analyze request - image: {image.filename}, content_type: {image.content_type}, cultures: {cultures}, use_multiple_wiki_pages: {use_multiple_wiki_pages}, model: {model_name}, session_id: {session_id}")
    
    try:
        # Validate image file
        if not image.content_type.startswith("image/"):
            logger.error(f"Invalid content type: {image.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Parse cultures
        culture_list = [c.strip() for c in cultures.split(",") if c.strip()]
        if not culture_list:
            logger.error(f"No cultures provided - cultures string was: '{cultures}'")
            raise HTTPException(status_code=400, detail="At least one culture must be provided")
        
        logger.info(f"Processing image for cultures: {culture_list}")
        
        # Validate model_name
        valid_models = ["qwen_vl", "pangea_vl", "llama_vl"]
        if model_name not in valid_models:
            logger.error(f"Invalid model: {model_name}")
            raise HTTPException(status_code=400, detail=f"Model must be one of: {', '.join(valid_models)}")
        
        # Read and process image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run CAIRE pipeline with optional session_id for caching
        result = pipeline.process_image(pil_image, culture_list, use_multiple_wiki_pages, model_name, session_id)
        
        # Format response
        response = AnalysisResponse(
            scores=[CultureScore(**score) for score in result["scores"]],
            wikipedia_pages=[WikipediaPage(**page) for page in result["wikipedia_pages"]],
            matched_entity=result["matched_entity"],
            image_path=image.filename or "uploaded_image",
            session_id=result.get("session_id")  # Include session_id for caching
        )
        
        logger.info(f"Analysis complete for {image.filename}, session_id: {result.get('session_id')}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.post("/api/analyze-with-predefined", response_model=AnalysisResponse)
async def analyze_with_predefined(
    image: UploadFile = File(...),
    list_name: str = Form(...),
    model_name: str = Form("qwen_vl"),
    session_id: Optional[str] = Form(None)
):
    """
    Analyze cultural relevance using a predefined culture list
    
    Args:
        image: Image file to analyze
        list_name: Name of predefined list (e.g., "top10_countries.pkl")
        model_name: VLM model to use for scoring (default: "qwen_vl")
                    Options: "qwen_vl", "pangea_vl", "llama_vl"
    
    Returns:
        Analysis results with cultural scores and Wikipedia pages
    """
    try:
        # Load predefined list
        list_path = DATA_PATH / list_name
        if list_path not in PREDEFINED_TARGET_LISTS:
            raise HTTPException(status_code=400, detail="Invalid predefined list name")
        
        with open(list_path, "rb") as f:
            culture_list = pickle.load(f)
        
        # Validate model_name
        valid_models = ["qwen_vl", "pangea_vl", "llama_vl"]
        if model_name not in valid_models:
            raise HTTPException(status_code=400, detail=f"Model must be one of: {', '.join(valid_models)}")
        
        # Read and process image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        logger.info(f"Processing image with predefined list: {list_name}, model: {model_name}, session_id: {session_id}")
        
        # Run CAIRE pipeline with optional session_id for caching
        result = pipeline.process_image(pil_image, culture_list, use_multiple_wiki_pages=False, model_name=model_name, session_id=session_id)
        
        # Format response  
        response = AnalysisResponse(
            scores=[CultureScore(**score) for score in result["scores"]],
            wikipedia_pages=[WikipediaPage(**page) for page in result["wikipedia_pages"]],
            matched_entity=result["matched_entity"],
            image_path=image.filename or "uploaded_image",
            session_id=result.get("session_id")  # Include session_id for caching
        )
        
        logger.info(f"Analysis complete for {image.filename}, session_id: {result.get('session_id')}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 