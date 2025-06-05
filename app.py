from flask import Flask, render_template
from dataset_selection import dataset_selection
from classification import classification
from clustring import clustering
from performance_evalution import performance_evaluation
from visuallization import visualization
from comparison import comparison
from improvement import improvement
import socket
import logging
from functools import lru_cache
from datetime import datetime
import gc
import traceback

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cache the improvement results to avoid recomputing
@lru_cache(maxsize=1)
def cached_improvement():
    try:
        logger.info("Computing improvement metrics")
        result = improvement()
        logger.info("Improvement metrics computed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in cached_improvement: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def index():
    logger.info("Accessing index page")
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return "Error rendering page", 500

@app.route('/dataset')
def dataset():
    logger.info("Accessing dataset page")
    try:
        summary, _, _ = dataset_selection()
        return render_template('dataset.html', summary=summary)
    except Exception as e:
        logger.error(f"Error in dataset route: {str(e)}")
        return "Error processing dataset", 500

@app.route('/classification')
def classification_route():
    logger.info("Accessing classification page")
    try:
        classification()
        return render_template('classification.html')
    except Exception as e:
        logger.error(f"Error in classification route: {str(e)}")
        return "Error processing classification", 500

@app.route('/clustering')
def clustering_route():
    logger.info("Accessing clustering page")
    try:
        silhouette, clusters, X_pca = clustering()
        return render_template('clustering.html', silhouette=silhouette)
    except Exception as e:
        logger.error(f"Error in clustering route: {str(e)}")
        return "Error processing clustering", 500

@app.route('/performance')
def performance_route():
    logger.info("Accessing performance page")
    try:
        results = performance_evaluation()
        return render_template('performance.html', results=results)
    except Exception as e:
        logger.error(f"Error in performance route: {str(e)}")
        return "Error processing performance evaluation", 500

@app.route('/visualization')
def visualization_route():
    logger.info("Accessing visualization page")
    try:
        visualization()
        return render_template('visualization.html')
    except Exception as e:
        logger.error(f"Error in visualization route: {str(e)}")
        return "Error processing visualization", 500

@app.route('/comparison')
def comparison_route():
    logger.info("Accessing comparison page")
    try:
        results_df, results_metrics = comparison()
        return render_template('comparison.html', 
                             results_df=results_df, 
                             results_metrics=results_metrics)
    except Exception as e:
        logger.error(f"Error in comparison route: {str(e)}")
        return "Error processing comparison", 500

@app.route('/improvement')
def show_improvement():
    logger.info("Accessing improvement page")
    try:
        # Use cached results
        original, improved, knn_params, nb_params, rf_params, n_components, top_features = cached_improvement()
        return render_template('improvement.html',
                             original=original,
                             improved=improved,
                             knn_params=knn_params,
                             nb_params=nb_params,
                             rf_params=rf_params,
                             n_components=n_components,
                             top_features=top_features)
    except Exception as e:
        logger.error(f"Error in improvement route: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error processing improvement metrics", 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "message": "Application is running"
        }, 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }, 500

def find_free_port(start_port=5000, max_attempts=100):
    """Find a free port to bind the Flask app"""
    port = start_port
    for _ in range(max_attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            sock.close()
            return port
        except socket.error:
            port += 1
            continue
    raise Exception(f"No free ports found between {start_port} and {port}")

if __name__ == '__main__':
    try:
        # Find a free port
        port = find_free_port()
        logger.info(f"Starting Flask app on port {port}")
        
        # Run the app
        app.run(
            debug=True,
            host='127.0.0.1',
            port=port,
            use_reloader=False  # Prevent duplicate runs in debug mode
        )
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up resources
        logger.info("Cleaning up resources")
        gc.collect()