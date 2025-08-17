"""
Configuration file for the Anime Recommendation System.

This file contains all the configurable parameters for the recommendation system.
You can modify these values to customize the behavior of the system.
"""

# Recommendation Parameters
DEFAULT_TOP_N_ANIME = 15
DEFAULT_TOP_N_CLUSTERS = 4
DEFAULT_K_PER_CLUSTER = 5
DEFAULT_MOST_POPULAR_CLUSTERS = 20

# Clustering Parameters
CLUSTERING_RANDOM_STATE = 42
DEFAULT_K_RANGE = (10, 201, 10)  # (start, stop, step) for K-means optimization

# Data Processing Parameters
RATING_FILL_VALUE = 0.0
UNKNOWN_EPISODES_VALUE = 0

# User Preference Weighting
FAVORITE_WEIGHT = 2.0  # How much more weight to give to favorite anime vs watched
USER_VECTOR_WEIGHT = 0.3  # Weight of user vector in entry-based recommendations
CONTENT_VECTOR_WEIGHT = 0.7  # Weight of content vector in entry-based recommendations

# API Configuration
JIKAN_API_DEBUG = True
JIKAN_REQUEST_DELAY = 0.5  # Delay between API requests to avoid rate limiting

# Visualization Parameters
DEFAULT_SAMPLE_SIZE = 1000
FIGURE_SIZE = (12, 8)
SCATTER_ALPHA = 0.6
SCATTER_SIZE = 30

# Search Parameters
DEFAULT_SEARCH_LIMIT = 10
SEARCH_CASE_SENSITIVE = False

# Dataset Configuration
DATASET_NAME = "CooperUnion/anime-recommendations-database"
ANIME_CSV_FILENAME = "anime.csv"
RATING_CSV_FILENAME = "rating.csv"  # For future use

# Feature Engineering
GENRE_SEPARATOR = ", "
MIN_MAX_FEATURE_NAMES = ["episodes_norm", "rating_norm", "members_norm"]

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance Settings
ENABLE_WARNINGS = False
NUMPY_RANDOM_SEED = 42

# File Paths (relative to project root)
CACHE_DIR = ".cache"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
LOGS_DIR = "logs"

# Model Persistence
SAVE_MODEL = True
MODEL_FILENAME = "anime_recommender_model.pkl"
SCALER_FILENAME = "feature_scalers.pkl"

# Validation Settings
MIN_CLUSTER_SIZE = 5  # Minimum number of anime per cluster
MAX_CLUSTERS = 200    # Maximum number of clusters to consider

# Error Handling
MAX_RETRIES = 3
RETRY_DELAY = 1.0

# Display Settings
MAX_DISPLAY_LENGTH = 50  # Maximum length for truncated text display
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

class RecommenderConfig:
    """Configuration class for the anime recommendation system."""
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with default values and optional overrides.
        
        Args:
            **kwargs: Configuration parameters to override
        """
        # Recommendation settings
        self.top_n_anime = kwargs.get('top_n_anime', DEFAULT_TOP_N_ANIME)
        self.top_n_clusters = kwargs.get('top_n_clusters', DEFAULT_TOP_N_CLUSTERS)
        self.k_per_cluster = kwargs.get('k_per_cluster', DEFAULT_K_PER_CLUSTER)
        self.most_popular_clusters = kwargs.get('most_popular_clusters', DEFAULT_MOST_POPULAR_CLUSTERS)
        
        # Clustering settings
        self.random_state = kwargs.get('random_state', CLUSTERING_RANDOM_STATE)
        self.k_range = kwargs.get('k_range', DEFAULT_K_RANGE)
        
        # Weighting settings
        self.favorite_weight = kwargs.get('favorite_weight', FAVORITE_WEIGHT)
        self.user_vector_weight = kwargs.get('user_vector_weight', USER_VECTOR_WEIGHT)
        self.content_vector_weight = kwargs.get('content_vector_weight', CONTENT_VECTOR_WEIGHT)
        
        # API settings
        self.jikan_debug = kwargs.get('jikan_debug', JIKAN_API_DEBUG)
        self.jikan_delay = kwargs.get('jikan_delay', JIKAN_REQUEST_DELAY)
        
        # Visualization settings
        self.sample_size = kwargs.get('sample_size', DEFAULT_SAMPLE_SIZE)
        self.figure_size = kwargs.get('figure_size', FIGURE_SIZE)
        
        # Search settings
        self.search_limit = kwargs.get('search_limit', DEFAULT_SEARCH_LIMIT)
        self.case_sensitive = kwargs.get('case_sensitive', SEARCH_CASE_SENSITIVE)
        
        # Dataset settings
        self.dataset_name = kwargs.get('dataset_name', DATASET_NAME)
        
        # Performance settings
        self.enable_warnings = kwargs.get('enable_warnings', ENABLE_WARNINGS)
        self.numpy_seed = kwargs.get('numpy_seed', NUMPY_RANDOM_SEED)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'top_n_anime': self.top_n_anime,
            'top_n_clusters': self.top_n_clusters,
            'k_per_cluster': self.k_per_cluster,
            'most_popular_clusters': self.most_popular_clusters,
            'random_state': self.random_state,
            'k_range': self.k_range,
            'favorite_weight': self.favorite_weight,
            'user_vector_weight': self.user_vector_weight,
            'content_vector_weight': self.content_vector_weight,
            'jikan_debug': self.jikan_debug,
            'jikan_delay': self.jikan_delay,
            'sample_size': self.sample_size,
            'figure_size': self.figure_size,
            'search_limit': self.search_limit,
            'case_sensitive': self.case_sensitive,
            'dataset_name': self.dataset_name,
            'enable_warnings': self.enable_warnings,
            'numpy_seed': self.numpy_seed
        }
    
    def validate(self):
        """Validate configuration parameters."""
        if self.top_n_anime <= 0:
            raise ValueError("top_n_anime must be positive")
        
        if self.top_n_clusters <= 0:
            raise ValueError("top_n_clusters must be positive")
        
        if self.k_per_cluster <= 0:
            raise ValueError("k_per_cluster must be positive")
        
        if self.most_popular_clusters <= 0:
            raise ValueError("most_popular_clusters must be positive")
        
        if not (0 <= self.user_vector_weight <= 1):
            raise ValueError("user_vector_weight must be between 0 and 1")
        
        if not (0 <= self.content_vector_weight <= 1):
            raise ValueError("content_vector_weight must be between 0 and 1")
        
        if abs(self.user_vector_weight + self.content_vector_weight - 1.0) > 1e-6:
            raise ValueError("user_vector_weight and content_vector_weight must sum to 1")
        
        return True
    
    def __str__(self):
        """String representation of configuration."""
        config_str = "AnimeRecommenderConfig:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str


# Create default configuration instance
default_config = RecommenderConfig()

# Environment-specific configurations
development_config = RecommenderConfig(
    jikan_debug=True,
    enable_warnings=True,
    top_n_anime=10,
    sample_size=500
)

production_config = RecommenderConfig(
    jikan_debug=False,
    enable_warnings=False,
    top_n_anime=20,
    sample_size=2000
)

testing_config = RecommenderConfig(
    top_n_anime=5,
    top_n_clusters=2,
    k_per_cluster=2,
    most_popular_clusters=3,
    sample_size=100,
    k_range=(5, 21, 5)  # Smaller range for faster testing
)


def get_config(environment='default'):
    """
    Get configuration based on environment.
    
    Args:
        environment: Configuration environment ('default', 'development', 'production', 'testing')
    
    Returns:
        RecommenderConfig: Configuration instance
    """
    configs = {
        'default': default_config,
        'development': development_config,
        'production': production_config,
        'testing': testing_config
    }
    
    if environment not in configs:
        raise ValueError(f"Unknown environment: {environment}. Available: {list(configs.keys())}")
    
    return configs[environment]


def load_config_from_file(filepath):
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to configuration file
    
    Returns:
        RecommenderConfig: Configuration instance
    """
    import json
    
    try:
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return RecommenderConfig(**config_dict)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def save_config_to_file(config, filepath):
    """
    Save configuration to a JSON file.
    
    Args:
        config: RecommenderConfig instance
        filepath: Path to save configuration file
    """
    import json
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
