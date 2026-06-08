import boto3
import os
from dotenv import load_dotenv

load_dotenv()

class S3Handler:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

    def upload_file(self, file_path, object_name=None):
        """Upload a file to an S3 bucket"""
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            return f"https://{self.bucket_name}.s3.amazonaws.com/{object_name}"
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            return None

    def download_file(self, object_name, download_path):
        """Download a file from an S3 bucket"""
        try:
            self.s3_client.download_file(self.bucket_name, object_name, download_path)
            return download_path
        except Exception as e:
            print(f"Error downloading from S3: {e}")
            return None

    def list_files(self):
        """List files in the S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            print(f"Error listing files in S3: {e}")
            return []
