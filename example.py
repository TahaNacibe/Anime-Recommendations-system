#!/usr/bin/env python3
"""
Example usage of the Anime Recommendation System.

This script demonstrates how to use the various features of the recommendation system.
"""

import asyncio
import sys
import os

# Add the current directory to Python path to import anime_recommender
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anime_recommender import AnimeRecommendationSystem


async def demonstrate_search():
    """Demonstrate the search functionality."""
    print("\n" + "="*60)
    print("SEARCHING FOR ANIME")
    print("="*60)
    
    recommender = AnimeRecommendationSystem()
    recommender.load_and_prepare_data()
    
    # Search examples
    queries = ["Naruto", "Attack", "Death Note", "One Piece"]
    
    for query in queries:
        print(f"\nSearching for '{query}':")
        results = recommender.search_anime(query, limit=3)
        if not results.empty:
            for _, anime in results.iterrows():
                print(f"  - {anime['name']} (ID: {anime['anime_id']}, Rating: {anime['rating']})")
        else:
            print(f"  No results found for '{query}'")


async def demonstrate_cold_start():
    """Demonstrate cold start recommendations."""
    print("\n" + "="*60)
    print("COLD START RECOMMENDATIONS")
    print("="*60)
    
    recommender = AnimeRecommendationSystem(
        top_n_anime=10,
        k_per_cluster=3,
        most_popular_clusters=5
    )
    
    recommender.load_and_prepare_data()
    recommender.train_clustering_model()
    
    print("\nGetting recommendations for new users...")
    recommendations = await recommender.get_cold_start_recommendations(return_entry=False)
    
    print("\nTop recommendations for new users:")
    for i, (_, anime) in enumerate(recommendations.head(10).iterrows(), 1):
        print(f"{i:2d}. {anime['name']}")
        print(f"    Genre: {anime['genre']}")
        print(f"    Type: {anime['type']}, Episodes: {anime['episodes']}")
        print(f"    Rating: {anime['rating']}, Members: {anime['members']:,}")
        print()


async def demonstrate_personalized_recommendations():
    """Demonstrate personalized recommendations."""
    print("\n" + "="*60)
    print("PERSONALIZED RECOMMENDATIONS")
    print("="*60)
    
    recommender = AnimeRecommendationSystem(top_n_anime=8)
    recommender.load_and_prepare_data()
    recommender.train_clustering_model()
    
    # Example user profile
    # These are popular anime IDs that likely exist in the dataset
    watched_anime = [
        5114,   # Fullmetal Alchemist: Brotherhood
        9253,   # Steins;Gate
        11061,  # Hunter x Hunter (2011)
        820,    # Ginga Eiyuu Densetsu
        28977   # Gintama°
    ]
    
    favorite_anime = [
        32281,  # Kimi no Na wa
        40748,  # Jujutsu Kaisen
        38524   # Shingeki no Kyojin Season 3 Part 2
    ]
    
    print("User Profile:")
    print("Watched anime IDs:", watched_anime)
    print("Favorite anime IDs:", favorite_anime)
    
    # Get anime information
    print("\nWatched anime details:")
    for anime_id in watched_anime[:3]:  # Show first 3 for brevity
        info = recommender.get_anime_info(anime_id)
        if info:
            print(f"  - {info['name']} (Rating: {info['rating']})")
    
    # Get global recommendations
    print("\nGlobal recommendations based on user preferences:")
    global_recs = await recommender.get_global_recommendations(
        watched_anime, favorite_anime, return_entry=False
    )
    
    for i, (_, anime) in enumerate(global_recs.head(5).iterrows(), 1):
        print(f"{i}. {anime['name']}")
        print(f"   Rating: {anime['rating']}, Members: {anime['members']:,}")
    
    # Get entry-based recommendations (if we have a valid anime ID)
    reference_anime_id = watched_anime[0]  # Use first watched anime as reference
    print(f"\nRecommendations similar to anime ID {reference_anime_id}:")
    
    try:
        entry_recs = await recommender.get_recommendations_based_on_entry(
            reference_anime_id, watched_anime, favorite_anime, return_entry=False
        )
        
        for i, (_, anime) in enumerate(entry_recs.head(5).iterrows(), 1):
            print(f"{i}. {anime['name']}")
            print(f"   Rating: {anime['rating']}, Genre: {anime['genre'][:50]}...")
            
    except Exception as e:
        print(f"Could not get entry-based recommendations: {e}")


async def demonstrate_clustering_info():
    """Demonstrate clustering information."""
    print("\n" + "="*60)
    print("CLUSTERING INFORMATION")
    print("="*60)
    
    recommender = AnimeRecommendationSystem()
    recommender.load_and_prepare_data()
    
    print("Finding optimal number of clusters...")
    optimal_k = recommender.find_optimal_clusters(k_range=(10, 51, 10))  # Smaller range for demo
    print(f"Optimal number of clusters: {optimal_k}")
    
    print("\nTraining clustering model...")
    recommender.train_clustering_model(n_clusters=optimal_k)
    
    # Show cluster distribution
    cluster_counts = recommender.anime_data['cluster'].value_counts().sort_index()
    print(f"\nCluster distribution (showing first 10 clusters):")
    for cluster_id, count in cluster_counts.head(10).items():
        print(f"Cluster {cluster_id}: {count} anime")
    
    # Show example anime from different clusters
    print(f"\nExample anime from different clusters:")
    for cluster_id in cluster_counts.index[:3]:
        cluster_anime = recommender.anime_data[
            recommender.anime_data['cluster'] == cluster_id
        ].head(2)
        
        print(f"\nCluster {cluster_id}:")
        for _, anime in cluster_anime.iterrows():
            print(f"  - {anime['name']} ({anime['genre']})")


async def interactive_demo():
    """Interactive demonstration allowing user input."""
    print("\n" + "="*60)
    print("INTERACTIVE DEMO")
    print("="*60)
    
    recommender = AnimeRecommendationSystem()
    
    print("Loading and preparing data... (this may take a moment)")
    recommender.load_and_prepare_data()
    recommender.train_clustering_model()
    
    print("\nSystem ready! You can now search for anime and get recommendations.")
    
    while True:
        print("\nOptions:")
        print("1. Search for anime")
        print("2. Get anime information by ID")
        print("3. Get cold start recommendations")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                query = input("Enter anime name to search: ").strip()
                if query:
                    results = recommender.search_anime(query, limit=10)
                    if not results.empty:
                        print(f"\nFound {len(results)} results:")
                        for i, (_, anime) in enumerate(results.iterrows(), 1):
                            print(f"{i:2d}. {anime['name']} (ID: {anime['anime_id']})")
                            print(f"     Genre: {anime['genre']}")
                            print(f"     Rating: {anime['rating']}, Type: {anime['type']}")
                    else:
                        print("No anime found with that name.")
                        
            elif choice == '2':
                try:
                    anime_id = int(input("Enter anime ID: ").strip())
                    info = recommender.get_anime_info(anime_id)
                    if info:
                        print(f"\nAnime Information:")
                        print(f"Name: {info['name']}")
                        print(f"Genre: {info['genre']}")
                        print(f"Type: {info['type']}")
                        print(f"Episodes: {info['episodes']}")
                        print(f"Rating: {info['rating']}")
                        print(f"Members: {info['members']:,}")
                        print(f"Cluster: {info['cluster']}")
                    else:
                        print("Anime not found with that ID.")
                except ValueError:
                    print("Please enter a valid anime ID (number).")
                    
            elif choice == '3':
                print("Getting cold start recommendations...")
                recs = await recommender.get_cold_start_recommendations(return_entry=False)
                print(f"\nTop {min(10, len(recs))} recommendations for new users:")
                for i, (_, anime) in enumerate(recs.head(10).iterrows(), 1):
                    print(f"{i:2d}. {anime['name']} (Rating: {anime['rating']})")
                    
            elif choice == '4':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


async def main():
    """Main function to run all demonstrations."""
    print("Anime Recommendation System - Demo")
    print("=" * 60)
    
    demos = [
        ("Search Functionality", demonstrate_search),
        ("Cold Start Recommendations", demonstrate_cold_start),
        ("Personalized Recommendations", demonstrate_personalized_recommendations),
        ("Clustering Information", demonstrate_clustering_info)
    ]
    
    # Run non-interactive demos first
    for name, demo_func in demos:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            await demo_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            continue
    
    # Ask if user wants interactive demo
    print("\n" + "="*60)
    response = input("Would you like to try the interactive demo? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        await interactive_demo()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())
