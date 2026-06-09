# Insere os dados dentro do MinIO

import boto3
from pathlib import Path

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

data_dir = Path("./data")

for file in data_dir.rglob("*"):
    if file.is_file():
        key = str(file.relative_to(data_dir))
        s3.upload_file(str(file), "bronze", key)
        print(f"✅ {key}")