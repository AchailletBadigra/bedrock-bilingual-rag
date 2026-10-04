import os
from pathlib import Path
import boto3
from dotenv import load_dotenv

load_dotenv()                      # reads your .env file
BUCKET = os.environ["BUCKET"]      # gets the bucket name from .env

def upload_folder(folder: str, prefix: str = "docs/") -> None:
    s3 = boto3.client("s3")        # connects Python to S3
    for file in Path(folder).glob("*.pdf"):
        key = prefix + file.name
        s3.upload_file(str(file), BUCKET, key)
        print(f"Uploaded: {file.name} -> s3://{BUCKET}/{key}")

if __name__ == "__main__":
    upload_folder("data")