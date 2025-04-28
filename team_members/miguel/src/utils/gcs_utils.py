"""
Google Cloud Storage (GCS) utility functions.
"""

from google.cloud import storage
import os
import logging

logger = logging.getLogger(__name__)

def authenticate_gcs():
    """
    Authenticate with Google Cloud Storage.
    
    Returns:
        storage.Client: Authenticated GCS client.
    """
    try:
        client = storage.Client()
        logger.info("Successfully authenticated with GCS")
        return client
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise

def list_bucket_contents(bucket_name="oasis-1-dataset-13635", prefix="", client=None):
    """
    List the contents of a GCS bucket.
    
    Args:
        bucket_name (str): Name of the GCS bucket.
        prefix (str, optional): Filter results to objects with names that begin with this prefix.
        client (storage.Client, optional): Authenticated GCS client.
        
    Returns:
        list: List of blob names in the bucket.
    """
    if client is None:
        client = authenticate_gcs()
    
    try:
        bucket = client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        blob_names = [blob.name for blob in blobs]
        return blob_names
    except Exception as e:
        logger.error(f"Error listing bucket contents: {str(e)}")
        return []

def download_blob(bucket_name, source_blob_name, destination_file_path, client=None):
    """
    Download a blob from GCS bucket to a local file.
    
    Args:
        bucket_name (str): Name of the GCS bucket.
        source_blob_name (str): Name of the blob to download.
        destination_file_path (str): Local path to save the downloaded file.
        client (storage.Client, optional): Authenticated GCS client.
        
    Returns:
        bool: True if download was successful, False otherwise.
    """
    if client is None:
        client = authenticate_gcs()
    
    try:
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
        
        # Download the blob
        blob.download_to_filename(destination_file_path)
        logger.info(f"Downloaded {source_blob_name} to {destination_file_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading blob {source_blob_name}: {str(e)}")
        return False
