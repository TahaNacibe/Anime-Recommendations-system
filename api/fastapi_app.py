#!/usr/bin/env python3
"""
FastAPI server for the Anime Recommendation System.

This API provides endpoints for getting anime recommendations, searching anime,
and managing user preferences.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from anime_recommender import AnimeRecommendationSystem
from config import get_config, RecommenderConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global recommender instance
recommender: Optional[AnimeRecommendationSystem] = None

# Pydantic models for API requests and responses
class AnimeInfo(BaseModel):
    """Anime information model."""
    anime_id: int
    name: str
    genre: Optional[str] = None
    type: Optional[str] = None
    episodes: Optional[int] = None
    rating: Optional[float] = None
    members: Optional[int] = None
    cluster: Optional[int] = None

class UserPreferences(BaseModel):
    """User preferences model."""
    watched_anime: List[int] = Field(default=[], description="List of watched anime IDs")
    favorite_anime: List[int] = Field(default=[], description="List of favorite anime IDs")
    
    @validator('watched_anime', 'favorite_anime')
    def validate_anime_ids(cls, v):
        if not isinstance(v, list):
            raise ValueError("Must be a list of anime IDs")
        for anime_id in v:
            if not isinstance(anime_id, int) or anime_id <= 0:
                raise ValueError("Anime IDs must be positive integers")
        return v

class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    user_preferences: UserPreferences
    reference_anime_id: Optional[int] = Field(None, description="Reference anime ID for entry-based recommendations")
    limit: Optional[int] = Field(15, ge=1, le=50, description="Number of recommendations to return")

class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[AnimeInfo]
    recommendation_type: str
    total_count: int
    user_preferences: UserPreferences

class SearchResponse(BaseModel):
    """Response model for search results."""
    results: List[AnimeInfo]
    total_count: int
    query: str

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    total_anime: Optional[int] = None

class SystemStatus(BaseModel):
    """System status response model."""
    status: str
    model_loaded: bool
    total_anime: Optional[int] = None
    total_clusters: Optional[int] = None
    config: Dict[str, Any]

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    logger.info("Starting up Anime Recommendation API...")
    await initialize_recommender()
    yield
    # Shutdown
    logger.info("Shutting down Anime Recommendation API...")

# Create FastAPI app
app = FastAPI(
    title="Anime Recommendation API",
    description="A machine learning-based anime recommendation system using K-means clustering and cosine similarity",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_recommender():
    """Initialize the recommendation system."""
    global recommender
    try:
        logger.info("Initializing recommendation system...")
        
        config = get_config('production')  # Use production config for API
        recommender = AnimeRecommendationSystem(
            top_n_anime=config.top_n_anime,
            top_n_clusters=config.top_n_clusters,
            k_per_cluster=config.k_per_cluster,
            most_popular_clusters=config.most_popular_clusters
        )
        
        logger.info("Loading and preparing data...")
        recommender.load_and_prepare_data()
        
        logger.info("Training clustering model...")
        recommender.train_clustering_model()
        
        logger.info("Recommendation system initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize recommendation system: {e}")
        raise

def get_recommender() -> AnimeRecommendationSystem:
    """Dependency to get the recommender instance."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation system not initialized")
    return recommender

def anime_to_dict(anime_row) -> Dict[str, Any]:
    """Convert pandas Series to dictionary with proper handling of NaN values."""
    anime_dict = anime_row.to_dict()
    
    # Handle NaN values and convert to appropriate types
    for key, value in anime_dict.items():
        if hasattr(value, 'isna') and value.isna():
            anime_dict[key] = None
        elif key in ['anime_id', 'episodes', 'members', 'cluster']:
            anime_dict[key] = int(value) if value is not None and not (hasattr(value, 'isna') and value.isna()) else None
        elif key in ['rating']:
            anime_dict[key] = float(value) if value is not None and not (hasattr(value, 'isna') and value.isna()) else None
    
    return anime_dict

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Anime Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global recommender
    
    model_loaded = recommender is not None
    total_anime = len(recommender.anime_data) if model_loaded and recommender.anime_data is not None else None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        total_anime=total_anime
    )

@app.get("/status", response_model=SystemStatus)
async def system_status(rec: AnimeRecommendationSystem = Depends(get_recommender)):
    """Get detailed system status."""
    config = get_config('production')
    
    return SystemStatus(
        status="healthy",
        model_loaded=True,
        total_anime=len(rec.anime_data),
        total_clusters=len(rec.anime_data['cluster'].unique()) if 'cluster' in rec.anime_data.columns else None,
        config=config.to_dict()
    )

@app.get("/search", response_model=SearchResponse)
async def search_anime(
    query: str = Query(..., description="Search query for anime name"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """Search for anime by name."""
    try:
        results_df = rec.search_anime(query, limit=limit)
        
        results = []
        for _, anime in results_df.iterrows():
            anime_dict = anime_to_dict(anime)
            results.append(AnimeInfo(**anime_dict))
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query=query
        )
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/anime/{anime_id}", response_model=AnimeInfo)
async def get_anime_info(
    anime_id: int,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """Get detailed information about a specific anime."""
    try:
        anime_info = rec.get_anime_info(anime_id)
        
        if anime_info is None:
            raise HTTPException(status_code=404, detail=f"Anime with ID {anime_id} not found")
        
        return AnimeInfo(**anime_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting anime info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get anime info: {str(e)}")

@app.post("/recommendations/cold-start", response_model=RecommendationResponse)
async def get_cold_start_recommendations(
    limit: int = Query(15, ge=1, le=50, description="Number of recommendations"),
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """Get cold start recommendations for new users."""
    try:
        # Temporarily set the limit
        original_limit = rec.top_n_anime
        rec.top_n_anime = limit
        
        recommendations_df = await rec.get_cold_start_recommendations(return_entry=False)
        
        # Restore original limit
        rec.top_n_anime = original_limit
        
        recommendations = []
        for _, anime in recommendations_df.iterrows():
            anime_dict = anime_to_dict(anime)
            recommendations.append(AnimeInfo(**anime_dict))
        
        return RecommendationResponse(
            recommendations=recommendations,
            recommendation_type="cold_start",
            total_count=len(recommendations),
            user_preferences=UserPreferences()
        )
        
    except Exception as e:
        logger.error(f"Error in cold start recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.post("/recommendations/global", response_model=RecommendationResponse)
async def get_global_recommendations(
    request: UserPreferences,
    limit: int = Query(15, ge=1, le=50, description="Number of recommendations"),
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """Get global recommendations based on user preferences."""
    try:
        if not request.watched_anime and not request.favorite_anime:
            raise HTTPException(status_code=400, detail="At least one watched or favorite anime is required")
        
        # Temporarily set the limit
        original_limit = rec.top_n_anime
        rec.top_n_anime = limit
        
        recommendations_df = await rec.get_global_recommendations(
            request.watched_anime, 
            request.favorite_anime, 
            return_entry=False
        )
        
        # Restore original limit
        rec.top_n_anime = original_limit
        
        recommendations = []
        for _, anime in recommendations_df.iterrows():
            anime_dict = anime_to_dict(anime)
            recommendations.append(AnimeInfo(**anime_dict))
        
        return RecommendationResponse(
            recommendations=recommendations,
            recommendation_type="global",
            total_count=len(recommendations),
            user_preferences=request
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in global recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.post("/recommendations/entry-based", response_model=RecommendationResponse)
async def get_entry_based_recommendations(
    request: RecommendationRequest,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """Get recommendations based on a specific anime entry."""
    try:
        if request.reference_anime_id is None:
            raise HTTPException(status_code=400, detail="reference_anime_id is required for entry-based recommendations")
        
        if not request.user_preferences.watched_anime and not request.user_preferences.favorite_anime:
            raise HTTPException(status_code=400, detail="At least one watched or favorite anime is required")
        
        # Temporarily set the limit
        original_limit = rec.top_n_anime
        rec.top_n_anime = request.limit
        
        recommendations_df = await rec.get_recommendations_based_on_entry(
            request.reference_anime_id,
            request.user_preferences.watched_anime,
            request.user_preferences.favorite_anime,
            return_entry=False
        )
        
        # Restore original limit
        rec.top_n_anime = original_limit
        
        recommendations = []
        for _, anime in recommendations_df.iterrows():
            anime_dict = anime_to_dict(anime)
            recommendations.append(AnimeInfo(**anime_dict))
        
        return RecommendationResponse(
            recommendations=recommendations,
            recommendation_type="entry_based",
            total_count=len(recommendations),
            user_preferences=request.user_preferences
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in entry-based recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.get("/clusters/{cluster_id}/anime", response_model=List[AnimeInfo])
async def get_anime_in_cluster(
    cluster_id: int,
    limit: int = Query(20, ge=1, le=100, description="Maximum number of anime to return"),
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """Get anime in a specific cluster."""
    try:
        if 'cluster' not in rec.anime_data.columns:
            raise HTTPException(status_code=500, detail="Clustering model not trained")
        
        cluster_anime = rec.anime_data[rec.anime_data['cluster'] == cluster_id].head(limit)
        
        if cluster_anime.empty:
            raise HTTPException(status_code=404, detail=f"No anime found in cluster {cluster_id}")
        
        results = []
        for _, anime in cluster_anime.iterrows():
            anime_dict = anime_to_dict(anime)
            results.append(AnimeInfo(**anime_dict))
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cluster anime: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cluster anime: {str(e)}")

@app.get("/clusters/stats", response_model=Dict[str, Any])
async def get_cluster_stats(
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """Get cluster statistics."""
    try:
        if 'cluster' not in rec.anime_data.columns:
            raise HTTPException(status_code=500, detail="Clustering model not trained")
        
        cluster_counts = rec.anime_data['cluster'].value_counts().sort_index()
        cluster_stats = rec.anime_data.groupby('cluster').agg({
            'rating': ['mean', 'std'],
            'members': ['mean', 'max', 'min'],
            'anime_id': 'count'
        }).round(2)
        
        # Flatten column names
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
        
        return {
            "total_clusters": len(cluster_counts),
            "cluster_sizes": cluster_counts.to_dict(),
            "cluster_statistics": cluster_stats.to_dict('index')
        }
        
    except Exception as e:
        logger.error(f"Error getting cluster stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cluster statistics: {str(e)}")

# Utility endpoints
@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the recommendation model (admin endpoint)."""
    async def reload_task():
        global recommender
        try:
            logger.info("Reloading recommendation model...")
            await initialize_recommender()
            logger.info("Model reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
    
    background_tasks.add_task(reload_task)
    return {"message": "Model reload initiated", "status": "in_progress"}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )
