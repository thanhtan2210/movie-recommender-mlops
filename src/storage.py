"""Read and write objects on Cloudflare R2 through its S3-compatible API."""
import hashlib
import os
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from src.config import load_r2_settings

SHA256_METADATA_KEY = "sha256"
STATS_FILE = "stats.json"


class RemoteConflictError(RuntimeError):
    """A remote object exists with different content; nothing was overwritten."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Storage:
    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_env(cls) -> "Storage":
        settings = load_r2_settings()
        client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            aws_access_key_id=settings.access_key_id,
            aws_secret_access_key=settings.secret_access_key,
        )
        return cls(client, settings.bucket)

    def list_objects(self, prefix: str = "") -> Dict[str, int]:
        """Key -> size in bytes, for every object under `prefix`."""
        objects = {}
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                objects[item["Key"]] = item["Size"]
        return objects

    def head(self, key: str) -> Optional[dict]:
        """Object metadata, or None if the object does not exist."""
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def download(self, key: str, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.client.download_file(self.bucket, key, path)

    def download_if_changed(self, key: str, path: str) -> bool:
        """Download unless a local file of the same size is already there. Returns True if downloaded."""
        remote = self.head(key)
        if remote is None:
            raise FileNotFoundError(f"s3://{self.bucket}/{key} does not exist")
        if os.path.exists(path) and os.path.getsize(path) == remote["ContentLength"]:
            return False
        self.download(key, path)
        return True

    def upload(self, path: str, key: str, sha256: Optional[str] = None) -> None:
        """Upload a file and record its sha256 as object metadata."""
        metadata = {SHA256_METADATA_KEY: sha256 or sha256_file(path)}
        self.client.upload_file(path, self.bucket, key, ExtraArgs={"Metadata": metadata})

    def remote_sha256(self, key: str) -> Optional[str]:
        """sha256 of a remote object, or None if it does not exist.

        Uses the metadata written by upload(); an object uploaded by other
        means is read back and hashed.
        """
        remote = self.head(key)
        if remote is None:
            return None
        recorded = remote.get("Metadata", {}).get(SHA256_METADATA_KEY)
        if recorded:
            return recorded
        digest = hashlib.sha256()
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        for block in iter(lambda: body.read(1 << 20), b""):
            digest.update(block)
        return digest.hexdigest()


def upload_directory(storage: Storage, local_dir: str, prefix: str, data_files: List[str]) -> Dict[str, str]:
    """Upload a processed directory without ever overwriting different content.

    For each data file: same sha256 on the remote -> skipped; missing ->
    uploaded; different -> RemoteConflictError, raised before anything is
    uploaded. stats.json is uploaded only when a data file was uploaded or
    when it is missing remotely. Returns {file name: "uploaded" | "skipped"}.
    """
    prefix = prefix.strip("/")
    local = {name: sha256_file(os.path.join(local_dir, name)) for name in data_files}
    remote = {name: storage.remote_sha256(f"{prefix}/{name}") for name in data_files}

    conflicts = [name for name in data_files if remote[name] is not None and remote[name] != local[name]]
    if conflicts:
        raise RemoteConflictError(
            f"s3://{storage.bucket}/{prefix}/ already holds different content for: {', '.join(conflicts)}. "
            "Nothing was uploaded or overwritten."
        )

    result = {}
    for name in data_files:
        if remote[name] is None:
            storage.upload(os.path.join(local_dir, name), f"{prefix}/{name}", local[name])
            result[name] = "uploaded"
        else:
            result[name] = "skipped"

    stats_key = f"{prefix}/{STATS_FILE}"
    if "uploaded" in result.values() or storage.head(stats_key) is None:
        storage.upload(os.path.join(local_dir, STATS_FILE), stats_key)
        result[STATS_FILE] = "uploaded"
    else:
        result[STATS_FILE] = "skipped"
    return result
