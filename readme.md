# Anime Recommendation System

A machine learning-based anime recommendation system using K-means clustering and cosine similarity to provide personalized anime recommendations.

## Features

- **Clustering-based recommendations**: Uses K-means clustering to group similar anime
- **Personalized recommendations**: Takes into account user's watched and favorite anime
- **Cold start support**: Provides recommendations for new users with no history
- **Multiple recommendation types**:
  - Entry-based: Recommendations based on a specific anime
  - Global: Recommendations based on user behavior patterns
  - Cold start: Popular anime from different clusters for new users
- **Search functionality**: Search anime by name
- **Visualization**: 2D cluster visualization using PCA

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/anime-recommendation-system.git
cd anime-recommendation-system
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import asyncio
from anime_recommender import AnimeRecommendationSystem

async def main():
    # Initialize the recommendation system
    recommender = AnimeRecommendationSystem(
        top_n_anime=15,
        top_n_clusters=4,
        k_per_cluster=5,
        most_popular_clusters=20
    )
    
    # Load and prepare data
    recommender.load_and_prepare_data()
    
    # Train the clustering model
    recommender.train_clustering_model()
    
    # Get cold start recommendations for new users
    cold_start = await recommender.get_cold_start_recommendations()
    print(cold_start[["name", "genre", "rating", "members"]].head(10))

# Run the example
asyncio.run(main())
```

### Advanced Usage

#### Search for Anime
```python
# Search for anime by name
search_results = recommender.search_anime("Naruto", limit=5)
print(search_results)
```

#### Get Personalized Recommendations
```python
# Define user's watched and favorite anime (using anime IDs)
watched_anime = [28977, 9969, 15335, 15417, 918]
favorite_anime = [30276, 245, 21]

# Get global recommendations based on user preferences
global_recs = await recommender.get_global_recommendations(
    watched_anime, favorite_anime, return_entry=False
)

# Get recommendations based on a specific anime
entry_recs = await recommender.get_recommendations_based_on_entry(
    918, watched_anime, favorite_anime, return_entry=False
)
```

#### Visualize Clusters
```python
# Plot anime clusters in 2D
recommender.plot_clusters_2d(sample_size=1000)
```

#### Get Anime Information
```python
# Get detailed information about a specific anime
anime_info = recommender.get_anime_info(918)
print(anime_info)
```

## Dataset

The system uses the "Anime Recommendations Database" from Kaggle, which contains:
- Anime ID, name, genre, type, episodes, rating, and member count
- Over 12,000 anime entries
- User ratings and preferences

The dataset is automatically downloaded when you first run the system.

## Algorithm Overview

1. **Data Preprocessing**:
   - Normalizes numerical features (episodes, rating, members)
   - Encodes categorical features (genres, types) using multi-label binarization
   - Creates feature vectors combining all attributes

2. **Clustering**:
   - Uses K-means clustering to group similar anime
   - Automatically determines optimal number of clusters using the elbow method
   - Assigns each anime to a cluster based on its features

3. **Recommendation Generation**:
   - **Entry-based**: Finds anime in the same cluster as the reference anime, then ranks using cosine similarity
   - **Global**: Identifies user's preferred clusters and recommends popular anime from those clusters
   - **Cold start**: Recommends popular anime from the most popular clusters

4. **Personalization**:
   - Creates user preference vectors from watched and favorite anime
   - Weights favorite anime more heavily than watched anime
   - Combines user preferences with content similarity

## Configuration

The system can be configured with the following parameters:

- `top_n_anime`: Number of recommendations to return (default: 15)
- `top_n_clusters`: Number of top clusters to consider for global recommendations (default: 4)
- `k_per_cluster`: Number of anime per cluster for cold start recommendations (default: 5)
- `most_popular_clusters`: Number of popular clusters to use for cold start (default: 20)

## API Integration

The system includes integration with the Jikan API for fetching additional anime metadata. This is optional and used when `return_entry=True` in recommendation methods.

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies

## Examples

Run the example script to see the system in action:

```bash
python anime_recommender.py
```

This will:
1. Download and prepare the dataset
2. Train the clustering model
3. Show example searches and recommendations
4. Generate visualization of clusters

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: "Anime Recommendations Database" from Kaggle by CooperUnion
- Jikan API for additional anime metadata
- scikit-learn for machine learning algorithms

## Future Enhancements

- Matrix factorization techniques (SVD, NMF)
- Deep learning approaches (autoencoders, neural collaborative filtering)
- Real-time user feedback integration
- Web interface for easy interaction
- More sophisticated user profiling
- Hybrid recommendation approaches
