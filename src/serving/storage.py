"""Download artifacts from Cloudflare R2 (S3-compatible)."""
import os
import shutil

import boto3

DB_PATH = "lancedb_movies"
DB_ZIP = "lancedb_movies.zip"


def has_r2_credentials() -> bool:
    return bool(os.environ.get('AWS_ACCESS_KEY_ID'))


def get_s3_client():
    return boto3.client('s3',
                        endpoint_url=os.environ.get('AWS_ENDPOINT_URL'),
                        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
                        )


def bucket_name() -> str:
    return os.environ.get('S3_BUCKET_NAME', 'movie-mlops')


def download_object(key: str, destination: str) -> None:
    get_s3_client().download_file(bucket_name(), key, destination)


def download_database(db_path: str = DB_PATH, zip_path: str = DB_ZIP) -> None:
    """Fetch lancedb_movies.zip from R2 and unpack it into `db_path`."""
    download_object(DB_ZIP, zip_path)
    shutil.unpack_archive(zip_path, db_path)
    os.remove(zip_path)
