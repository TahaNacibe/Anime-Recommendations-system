import random
import numpy as np
import pandas as pd
import kagglehub as kb
import asyncio
import jikan4snek
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from kneed import KneeLocator
import warnings
import os

# Import configuration
from config import get_config, RecommenderConfig


class AnimeRecommendationSystem:
    """
    Anime recommendation system using K-means clustering and cosine similarity.
    """
    
    def __init__(self, config=None, environment='default'):
        """
        Initialize the recommendation system with configuration.
        
        Args:
            config: RecommenderConfig instance (optional)
            environment: Configuration environment if config is None
        """
        # Load configuration
        if config is None:
            self.config = get_config(environment)
        else:
            self.config = config
        
        # Validate configuration
        self.config.validate()
        
        # Set up warnings
        if not self.config.enable_warnings:
            warnings.filterwarnings('ignore')
        
        # Set random seeds for reproducibility
        random.seed(self.config.random_state)
        np.random.seed(self.config.numpy_seed)
        
        # Data storage
        self.anime_data = None
        self.feature_vectors = None
        self.k_means = None
        self.genre_mlb = None
        self.type_mlb = None
        self.episodes_scaler = None
        self.rating_scaler = None
        self.members_scaler = None
        
        # Create necessary directories
        self._create_directories()
        
        print(f"Initialized AnimeRecommendationSystem with {environment} configuration")
    
    def _create_directories(self):
        """Create necessary directories for caching, models, plots, and logs."""
        from config import CACHE_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR
        
        directories = [CACHE_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_and_prepare_data(self):
        """
        Load and prepare the anime dataset for clustering.
        """
        from config import ANIME_CSV_FILENAME
        
        # Load the anime dataset
        file_path = kb.dataset_download(self.config.dataset_name)
        print(f"Path to dataset files: {file_path}")
        
        # Read the data
        self.anime_data = pd.read_csv(f"{file_path}/{ANIME_CSV_FILENAME}")
        print(f"Loaded {len(self.anime_data)} anime entries")
        
        # Normalize the data
        self._normalize_data()
        
        # Encode categorical features
        self._encode_features()
        
        # Create feature vectors
        self._create_feature_vectors()
        
        print("Data preparation completed!")
    
    def _normalize_data(self):
        """
        Normalize numerical features using configuration values.
        """
        from config import UNKNOWN_EPISODES_VALUE, RATING_FILL_VALUE, MIN_MAX_FEATURE_NAMES
        
        # Handle unknown episodes
        self.anime_data.loc[self.anime_data["episodes"] == "Unknown", "episodes"] = UNKNOWN_EPISODES_VALUE
        
        # Fill rating empty cells
        self.anime_data["rating"] = pd.to_numeric(self.anime_data["rating"], errors="coerce").fillna(RATING_FILL_VALUE)
        
        # Create scalers
        self.episodes_scaler = MinMaxScaler()
        self.rating_scaler = MinMaxScaler()
        self.members_scaler = MinMaxScaler()
        
        # Scale data using the feature names from config
        self.anime_data["episodes_norm"] = self.episodes_scaler.fit_transform(self.anime_data[["episodes"]])
        self.anime_data["rating_norm"] = self.rating_scaler.fit_transform(self.anime_data[["rating"]])
        self.anime_data["members_norm"] = self.members_scaler.fit_transform(self.anime_data[["members"]])
    
    def _encode_features(self):
        """
        Encode categorical features (genres and types).
        """
        from config import GENRE_SEPARATOR
        
        # Encode genres
        self.genre_mlb = MultiLabelBinarizer()
        self.anime_data["genre_list"] = self.anime_data["genre"].fillna("").apply(
            lambda x: x.split(GENRE_SEPARATOR)
        )
        genre_encoded = self.genre_mlb.fit_transform(self.anime_data["genre_list"])
        genre_df = pd.DataFrame(genre_encoded, columns=self.genre_mlb.classes_)
        
        # Encode types
        self.type_mlb = MultiLabelBinarizer()
        type_encoded = self.type_mlb.fit_transform(self.anime_data["type"].fillna("").apply(lambda x: [x]))
        type_df = pd.DataFrame(type_encoded, columns=self.type_mlb.classes_)
        
        self.genre_df = genre_df
        self.type_df = type_df
    
    def _create_feature_vectors(self):
        """
        Create feature vectors by combining all encoded features.
        """
        from config import MIN_MAX_FEATURE_NAMES
        
        self.feature_vectors = np.hstack([
            self.genre_df.values,
            self.type_df.values,
            self.anime_data[MIN_MAX_FEATURE_NAMES].values
        ])
    
    def find_optimal_clusters(self):
        """
        Find optimal number of clusters using the elbow method.
        
        Returns:
            int: Optimal number of clusters
        """
        from config import MAX_CLUSTERS, MIN_CLUSTER_SIZE
        
        k_range = self.config.k_range
        inertia = []
        ks = list(range(*k_range))
        
        # Ensure we don't exceed maximum clusters or go below minimum
        ks = [k for k in ks if MIN_CLUSTER_SIZE <= k <= MAX_CLUSTERS]
        
        print(f"Finding optimal number of clusters in range: {min(ks)}-{max(ks)}...")
        for k in ks:
            k_means = KMeans(n_clusters=k, random_state=self.config.random_state)
            k_means.fit(self.feature_vectors)
            inertia.append(k_means.inertia_)
        
        # Calculate optimal k using knee locator
        knee = KneeLocator(ks, inertia, curve="convex", direction="decreasing")
        optimal_k = knee.knee
        
        if optimal_k is None:
            optimal_k = ks[len(ks)//2]  # Fallback to middle value
            print(f"Could not find clear elbow, using middle value: {optimal_k}")
        else:
            print(f"Optimal number of clusters: {optimal_k}")
        
        return optimal_k
    
    def train_clustering_model(self, n_clusters=None):
        """
        Train the K-means clustering model.
        
        Args:
            n_clusters: Number of clusters (if None, will find optimal)
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        print(f"Training K-means model with {n_clusters} clusters...")
        self.k_means = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
        self.k_means.fit(self.feature_vectors)
        
        # Add cluster labels to anime data
        self.anime_data["cluster"] = self.k_means.labels_
        print("Clustering model trained successfully!")
        
        # Save model if configured to do so
        self._save_model_if_configured()
    
    def _save_model_if_configured(self):
        """Save the trained model if SAVE_MODEL is True."""
        from config import SAVE_MODEL, MODELS_DIR, MODEL_FILENAME, SCALER_FILENAME
        import pickle
        
        if SAVE_MODEL:
            # Save the K-means model
            model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'k_means': self.k_means,
                    'genre_mlb': self.genre_mlb,
                    'type_mlb': self.type_mlb,
                    'feature_vectors': self.feature_vectors,
                    'anime_data': self.anime_data
                }, f)
            
            # Save the scalers
            scaler_path = os.path.join(MODELS_DIR, SCALER_FILENAME)
            with open(scaler_path, 'wb') as f:
                pickle.dump({
                    'episodes_scaler': self.episodes_scaler,
                    'rating_scaler': self.rating_scaler,
                    'members_scaler': self.members_scaler
                }, f)
            
            print(f"Model saved to {model_path}")
            print(f"Scalers saved to {scaler_path}")
    
    def get_cluster_entries(self, anime_id):
        """
        Get all anime entries in the same cluster as the given anime.
        
        Args:
            anime_id: ID of the anime
        
        Returns:
            DataFrame: Anime entries in the same cluster
        """
        anime_index = self.anime_data[self.anime_data["anime_id"] == anime_id].index[0]
        target_cluster = self.anime_data.loc[anime_index, "cluster"]
        
        recommendations = self.anime_data[
            (self.anime_data["cluster"] == target_cluster) &
            (self.anime_data["anime_id"] != anime_id)
        ]
        
        return recommendations
    
    def get_user_vector(self, watched_ids_list, favorite_ids_list):
        """
        Create a user preference vector based on watched and favorite anime.
        
        Args:
            watched_ids_list: List of watched anime IDs
            favorite_ids_list: List of favorite anime IDs
        
        Returns:
            np.array: User preference vector
        """
        watched_indices = self.anime_data[self.anime_data["anime_id"].isin(watched_ids_list)].index
        favorite_indices = self.anime_data[self.anime_data["anime_id"].isin(favorite_ids_list)].index
        
        # Weight favorites more heavily using config values
        user_vector = (
            self.feature_vectors[favorite_indices].mean(axis=0) * self.config.favorite_weight +
            self.feature_vectors[watched_indices].mean(axis=0)
        ) * self.config.user_vector_weight
        
        return user_vector.reshape(1, -1)
    
    def dataframe_to_ids(self, df):
        """
        Convert DataFrame to list of anime IDs.
        
        Args:
            df: DataFrame containing anime data
        
        Returns:
            list: List of anime IDs
        """
        return df["anime_id"].tolist()
    
    async def get_anime_list(self, data_frame):
        """
        Get anime data from Jikan API with retry logic.
        
        Args:
            data_frame: DataFrame containing anime data
        
        Returns:
            int: Number of anime entries processed
        """
        from config import MAX_RETRIES, RETRY_DELAY
        
        anime_list_ids = self.dataframe_to_ids(data_frame)
        jikan = jikan4snek.Jikan4SNEK(debug=self.config.jikan_debug)
        
        successful_requests = 0
        
        for anime_id in anime_list_ids:
            for attempt in range(MAX_RETRIES):
                try:
                    res = await jikan.get(anime_id).anime()
                    if res and 'data' in res:
                        successful_requests += 1
                    
                    # Add delay to avoid rate limiting
                    await asyncio.sleep(self.config.jikan_delay)
                    break
                    
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        print(f"Retry {attempt + 1} for anime {anime_id}: {str(e)}")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        print(f"Failed to get data for anime {anime_id} after {MAX_RETRIES} attempts")
        
        return successful_requests
    
    async def get_recommendations_based_on_entry(self, anime_id, watched_ids_list, favorite_ids_list, return_entry=False):
        """
        Get recommendations based on a specific anime entry and user preferences.
        
        Args:
            anime_id: ID of the reference anime
            watched_ids_list: List of watched anime IDs
            favorite_ids_list: List of favorite anime IDs
            return_entry: Whether to return API data or DataFrame
        
        Returns:
            DataFrame or int: Recommendations or number of entries
        """
        # Get same cluster entries
        same_cluster_entries = self.get_cluster_entries(anime_id)
        cluster_indices = same_cluster_entries.index
        cluster_vectors = self.feature_vectors[cluster_indices]
        
        # Get input anime vector
        anime_idx = self.anime_data[self.anime_data["anime_id"] == anime_id].index[0]
        input_vector = self.feature_vectors[anime_idx].reshape(1, -1)
        
        # Get user preference vector
        user_vector = self.get_user_vector(watched_ids_list, favorite_ids_list)
        final_vector = (self.config.user_vector_weight * user_vector + 
                       self.config.content_vector_weight * input_vector)
        
        # Compute cosine similarity
        similarities = cosine_similarity(final_vector, cluster_vectors).flatten()
        
        # Rank results
        ranked_indices = cluster_indices[np.argsort(similarities)[::-1]]
        top_n_ids = self.anime_data.loc[ranked_indices, "anime_id"].values[:self.config.top_n_anime]
        
        # Remove already watched/favorite anime
        top_n_ids = [id for id in top_n_ids if id not in watched_ids_list + favorite_ids_list + [anime_id]]
        
        # Get anime items
        top_n_items = self.anime_data[self.anime_data["anime_id"].isin(top_n_ids)]
        
        return await self.get_anime_list(top_n_items) if return_entry else top_n_items
    
    async def get_global_recommendations(self, watched_anime_ids, favorite_anime_ids, return_entry=False):
        """
        Get global recommendations based on user behavior patterns.
        
        Args:
            watched_anime_ids: List of watched anime IDs
            favorite_anime_ids: List of favorite anime IDs
            return_entry: Whether to return API data or DataFrame
        
        Returns:
            DataFrame or int: Recommendations or number of entries
        """
        # Remove duplicates
        anime_ids = set(watched_anime_ids + favorite_anime_ids)
        
        # Get clusters in user lists
        anime_entries = self.anime_data[self.anime_data["anime_id"].isin(anime_ids)]
        cluster_ids = Counter(anime_entries["cluster"])
        
        # Get top clusters
        top_clusters = [cluster for cluster, _ in cluster_ids.most_common(6)]
        selected_clusters = random.sample(
            top_clusters, 
            k=min(self.config.top_n_clusters, len(top_clusters))
        )
        
        # Get candidate anime
        candidate_anime = self.anime_data[self.anime_data["cluster"].isin(selected_clusters)]
        candidate_anime = candidate_anime[~candidate_anime["anime_id"].isin(anime_ids)]
        
        # Sort and get recommendations
        recommended = candidate_anime.head(self.config.top_n_anime)
        
        return await self.get_anime_list(recommended) if return_entry else recommended
    
    async def get_cold_start_recommendations(self, return_entry=False):
        """
        Get cold start recommendations for new users.
        
        Args:
            return_entry: Whether to return API data or DataFrame
        
        Returns:
            DataFrame or int: Recommendations or number of entries
        """
        # Get most popular clusters by average member count
        popular_clusters = (
            self.anime_data.groupby("cluster")["members"]
            .mean()
            .sort_values(ascending=False)
            .head(self.config.most_popular_clusters)
            .index
        )
        
        # Collect top-k anime from each cluster
        clusters_anime_list = []
        for cluster_id in popular_clusters:
            cluster_anime = self.anime_data[self.anime_data["cluster"] == cluster_id]
            top_k_anime = cluster_anime.sort_values(by="members", ascending=False).head(self.config.k_per_cluster)
            clusters_anime_list.append(top_k_anime)
        
        # Concatenate all into one DataFrame
        recommended = pd.concat(clusters_anime_list).reset_index(drop=True)
        
        return await self.get_anime_list(recommended) if return_entry else recommended
    
    def plot_clusters_2d(self, save_plot=False):
        """
        Plot anime clusters in 2D using PCA.
        
        Args:
            save_plot: Whether to save the plot to file
        """
        from config import PLOTS_DIR
        
        sample_size = self.config.sample_size
        
        # Sample data if needed
        if sample_size is not None and sample_size < len(self.feature_vectors):
            np.random.seed(self.config.numpy_seed)
            indices = np.random.choice(len(self.feature_vectors), sample_size, replace=False)
            features_sample = self.feature_vectors[indices]
            labels_sample = np.array(self.anime_data["cluster"])[indices]
        else:
            features_sample = self.feature_vectors
            labels_sample = self.anime_data["cluster"]
        
        # Reduce dimensions using PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(features_sample)
        
        # Plot using config parameters
        plt.figure(figsize=self.config.figure_size)
        from config import SCATTER_ALPHA, SCATTER_SIZE
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], 
            c=labels_sample, cmap='tab20', 
            alpha=SCATTER_ALPHA, s=SCATTER_SIZE
        )
        plt.title("Anime Clusters Visualization (PCA)", fontsize=16)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.tight_layout()
        
        if save_plot:
            from config import DATETIME_FORMAT
            import datetime
            
            timestamp = datetime.datetime.now().strftime(DATETIME_FORMAT.replace(":", "-").replace(" ", "_"))
            plot_path = os.path.join(PLOTS_DIR, f"clusters_2d_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.show()
    
    def get_anime_info(self, anime_id):
        """
        Get information about a specific anime.
        
        Args:
            anime_id: ID of the anime
        
        Returns:
            dict: Anime information
        """
        anime_row = self.anime_data[self.anime_data["anime_id"] == anime_id]
        if anime_row.empty:
            return None
        
        anime_info = anime_row.iloc[0]
        return {
            "anime_id": anime_info["anime_id"],
            "name": anime_info["name"],
            "genre": anime_info["genre"],
            "type": anime_info["type"],
            "episodes": anime_info["episodes"],
            "rating": anime_info["rating"],
            "members": anime_info["members"],
            "cluster": anime_info.get("cluster", "Not clustered yet")
        }
    
    def search_anime(self, query, limit=None):
        """
        Search for anime by name.
        
        Args:
            query: Search query
            limit: Maximum number of results (uses config default if None)
        
        Returns:
            DataFrame: Search results
        """
        if self.anime_data is None:
            print("Data not loaded yet. Please run load_and_prepare_data() first.")
            return pd.DataFrame()
        
        if limit is None:
            limit = self.config.search_limit
        
        # Search based on case sensitivity setting
        mask = self.anime_data["name"].str.contains(
            query, case=self.config.case_sensitive, na=False
        )
        results = self.anime_data[mask].head(limit)
        
        return results[["anime_id", "name", "genre", "type", "rating", "members"]]
    
    def get_cluster_summary(self):
        """
        Get a summary of clusters in the dataset.
        
        Returns:
            DataFrame: Cluster summary statistics
        """
        if self.k_means is None:
            print("Model not trained yet. Please run train_clustering_model() first.")
            return pd.DataFrame()
        
        cluster_summary = self.anime_data.groupby('cluster').agg({
            'anime_id': 'count',
            'rating': 'mean',
            'members': 'mean',
            'episodes': 'mean'
        }).round(2)
        
        cluster_summary.columns = ['anime_count', 'avg_rating', 'avg_members', 'avg_episodes']
        cluster_summary = cluster_summary.sort_values('avg_members', ascending=False)
        
        return cluster_summary


# Example usage and testing functions
async def main():
    """
    Example usage of the AnimeRecommendationSystem with different configurations.
    """
    print("=== Testing Different Configuration Environments ===\n")
    
    # Test with development configuration
    print("1. Development Configuration:")
    dev_recommender = AnimeRecommendationSystem(environment='development')
    print(f"Config: top_n_anime={dev_recommender.config.top_n_anime}, "
          f"sample_size={dev_recommender.config.sample_size}")
    
    # Load and prepare data
    dev_recommender.load_and_prepare_data()
    
    # Train the clustering model
    dev_recommender.train_clustering_model()
    
    # Example: Search for anime
    print("\nSearching for 'Naruto':")
    search_results = dev_recommender.search_anime("Naruto", limit=5)
    print(search_results)
    
    # Example: Get cluster summary
    print("\nCluster Summary:")
    cluster_summary = dev_recommender.get_cluster_summary()
    print(cluster_summary.head())
    
    # Example: Get cold start recommendations
    print("\nCold start recommendations:")
    cold_start = await dev_recommender.get_cold_start_recommendations(return_entry=False)
    print(cold_start[["name", "genre", "rating", "members"]].head(10))
    
    # Example: Get recommendations based on user preferences
    watched = [28977, 9969, 15335, 15417, 918]  # Example watched anime IDs
    favorites = [30276, 245, 21]  # Example favorite anime IDs
    
    print("\nGlobal recommendations based on user preferences:")
    global_recs = await dev_recommender.get_global_recommendations(watched, favorites, return_entry=False)
    print(global_recs[["name", "genre", "rating", "members"]].head(10))
    
    # Example: Get recommendations based on specific anime
    print("\nRecommendations based on specific anime (ID: 918):")
    entry_recs = await dev_recommender.get_recommendations_based_on_entry(918, watched, favorites, return_entry=False)
    print(entry_recs[["name", "genre", "rating", "members"]].head(10))
    
    # Plot clusters (optional - requires matplotlib)
    print("\nPlotting clusters...")
    dev_recommender.plot_clusters_2d(save_plot=True)
    
    # Test with custom configuration
    print("\n\n2. Custom Configuration:")
    custom_config = RecommenderConfig(
        top_n_anime=25,
        sample_size=1500,
        jikan_debug=False,
        search_limit=15
    )
    
    custom_recommender = AnimeRecommendationSystem(config=custom_config)
    print(f"Custom config: top_n_anime={custom_recommender.config.top_n_anime}, "
          f"sample_size={custom_recommender.config.sample_size}")
    
    print("\n=== Configuration Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())