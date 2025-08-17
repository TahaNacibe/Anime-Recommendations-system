#!/usr/bin/env python3
"""
Python client for the Anime Recommendation API.

This module provides a convenient Python interface for interacting with the
FastAPI-based anime recommendation service.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import json


class AnimeRecommendationClient:
    """
    Client for the Anime Recommendation API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional arguments for the request
            
        Returns:
            Dict containing the response data
            
        Raises:
            Exception: If the request fails
        """
        session = self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Dict containing health information
        """
        return await self._request("GET", "/health")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get detailed system status.
        
        Returns:
            Dict containing system status information
        """
        return await self._request("GET", "/status")
    
    async def search_anime(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for anime by name.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Dict containing search results
        """
        params = {"query": query, "limit": limit}
        return await self._request("GET", "/search", params=params)
    
    async def get_anime_info(self, anime_id: int) -> Dict[str, Any]:
        """
        Get information about a specific anime.
        
        Args:
            anime_id: ID of the anime
            
        Returns:
            Dict containing anime information
        """
        return await self._request("GET", f"/anime/{anime_id}")
    
    async def get_cold_start_recommendations(self, limit: int = 15) -> Dict[str, Any]:
        """
        Get cold start recommendations for new users.
        
        Args:
            limit: Number of recommendations to get
            
        Returns:
            Dict containing recommendations
        """
        params = {"limit": limit}
        return await self._request("POST", "/recommendations/cold-start", params=params)
    
    async def get_global_recommendations(
        self, 
        watched_anime: List[int], 
        favorite_anime: List[int] = None, 
        limit: int = 15
    ) -> Dict[str, Any]:
        """
        Get global recommendations based on user preferences.
        
        Args:
            watched_anime: List of watched anime IDs
            favorite_anime: List of favorite anime IDs
            limit: Number of recommendations to get
            
        Returns:
            Dict containing recommendations
        """
        if favorite_anime is None:
            favorite_anime = []
        
        data = {
            "watched_anime": watched_anime,
            "favorite_anime": favorite_anime
        }
        params = {"limit": limit}
        
        return await self._request(
            "POST", 
            "/recommendations/global", 
            json=data, 
            params=params
        )
    
    async def get_entry_based_recommendations(
        self,
        reference_anime_id: int,
        watched_anime: List[int],
        favorite_anime: List[int] = None,
        limit: int = 15
    ) -> Dict[str, Any]:
        """
        Get recommendations based on a specific anime entry.
        
        Args:
            reference_anime_id: ID of the reference anime
            watched_anime: List of watched anime IDs
            favorite_anime: List of favorite anime IDs
            limit: Number of recommendations to get
            
        Returns:
            Dict containing recommendations
        """
        if favorite_anime is None:
            favorite_anime = []
        
        data = {
            "user_preferences": {
                "watched_anime": watched_anime,
                "favorite_anime": favorite_anime
            },
            "reference_anime_id": reference_anime_id,
            "limit": limit
        }
        
        return await self._request("POST", "/recommendations/entry-based", json=data)
    
    async def get_cluster_anime(self, cluster_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get anime in a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            limit: Maximum number of anime to return
            
        Returns:
            List of anime in the cluster
        """
        params = {"limit": limit}
        return await self._request("GET", f"/clusters/{cluster_id}/anime", params=params)
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get cluster statistics.
        
        Returns:
            Dict containing cluster statistics
        """
        return await self._request("GET", "/clusters/stats")
    
    async def reload_model(self) -> Dict[str, Any]:
        """
        Reload the recommendation model.
        
        Returns:
            Dict containing reload status
        """
        return await self._request("POST", "/reload-model")


# Convenience functions for common operations
async def quick_search(query: str, base_url: str = "http://localhost:8000") -> List[Dict[str, Any]]:
    """
    Quick search for anime.
    
    Args:
        query: Search query
        base_url: API base URL
        
    Returns:
        List of anime results
    """
    async with AnimeRecommendationClient(base_url) as client:
        response = await client.search_anime(query)
        return response.get("results", [])


async def quick_recommendations(
    watched_anime: List[int], 
    favorite_anime: List[int] = None,
    base_url: str = "http://localhost:8000"
) -> List[Dict[str, Any]]:
    """
    Quick global recommendations.
    
    Args:
        watched_anime: List of watched anime IDs
        favorite_anime: List of favorite anime IDs
        base_url: API base URL
        
    Returns:
        List of recommendations
    """
    async with AnimeRecommendationClient(base_url) as client:
        response = await client.get_global_recommendations(watched_anime, favorite_anime)
        return response.get("recommendations", [])


async def quick_cold_start(base_url: str = "http://localhost:8000") -> List[Dict[str, Any]]:
    """
    Quick cold start recommendations.
    
    Args:
        base_url: API base URL
        
    Returns:
        List of recommendations
    """
    async with AnimeRecommendationClient(base_url) as client:
        response = await client.get_cold_start_recommendations()
        return response.get("recommendations", [])


# Example usage and testing
async def main():
    """Example usage of the API client."""
    base_url = "http://localhost:8000"
    
    async with AnimeRecommendationClient(base_url) as client:
        try:
            # Check health
            print("Checking API health...")
            health = await client.health_check()
            print(f"Health status: {health}")
            
            # Search for anime
            print("\nSearching for 'Naruto'...")
            search_results = await client.search_anime("Naruto", limit=5)
            print(f"Found {len(search_results['results'])} results:")
            for anime in search_results["results"]:
                print(f"  - {anime['name']} (ID: {anime['anime_id']}, Rating: {anime['rating']})")
            
            # Get cold start recommendations
            print("\nGetting cold start recommendations...")
            cold_start = await client.get_cold_start_recommendations(limit=5)
            print(f"Cold start recommendations ({len(cold_start['recommendations'])}):")
            for anime in cold_start["recommendations"]:
                print(f"  - {anime['name']} (Rating: {anime['rating']})")
            
            # Get global recommendations (using some example anime IDs)
            watched = [5114, 9253, 11061]  # Example IDs
            favorites = [32281, 40748]     # Example IDs
            
            print(f"\nGetting global recommendations...")
            print(f"Watched: {watched}")
            print(f"Favorites: {favorites}")
            
            global_recs = await client.get_global_recommendations(watched, favorites, limit=5)
            print(f"Global recommendations ({len(global_recs['recommendations'])}):")
            for anime in global_recs["recommendations"]:
                print(f"  - {anime['name']} (Rating: {anime['rating']})")
            
            # Get entry-based recommendations
            reference_id = watched[0]
            print(f"\nGetting recommendations similar to anime ID {reference_id}...")
            
            entry_recs = await client.get_entry_based_recommendations(
                reference_id, watched, favorites, limit=5
            )
            print(f"Entry-based recommendations ({len(entry_recs['recommendations'])}):")
            for anime in entry_recs["recommendations"]:
                print(f"  - {anime['name']} (Rating: {anime['rating']})")
            
            # Get system status
            print("\nGetting system status...")
            status = await client.get_system_status()
            print(f"Total anime in system: {status['total_anime']}")
            print(f"Total clusters: {status['total_clusters']}")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
