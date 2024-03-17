import json
from io import BytesIO

import boto3

credential = json.load(open("others/xueyuan_minio.json", "r"))
credential1 = json.load(open("others/shahe_minio.json", "r"))
s3_client = boto3.client('s3', aws_access_key_id=credential['accessKey'], aws_secret_access_key=credential['secretKey'],
                         endpoint_url=credential['url'])
s3_client_shahe = boto3.client('s3', aws_access_key_id=credential1['accessKey'], aws_secret_access_key=credential1['secretKey'],
                               endpoint_url=credential1['url'])

bucket_name = 'unipde'


def download_unipde(localpath, remotepath):
    s3_client.download_file(Bucket=bucket_name, Key=remotepath, Filename=localpath)


def download(bucket, localpath, remotepath):
    s3_client.download_file(Bucket=bucket, Key=remotepath, Filename=localpath)


def upload_unipde(localpath, remotepath):
    s3_client.upload_file(localpath, bucket_name, remotepath)


def upload_unipde_shahe(localpath, remotepath):
    s3_client_shahe.upload_file(localpath, bucket_name, remotepath)


def upload(bucket, localpath, remotepath):
    s3_client.upload_file(Filename=localpath, Bucket=bucket, Key=remotepath)


def get_client():
    return s3_client


def fileobj(bucket, key):
    """
    Yields a file object from the filename at {bucket}/{key}

    Args:
        bucket (str): Name of the S3 bucket where you model is stored
        key (str): Relative path from the base of your bucket, including the filename and extension of the object to be retrieved.
    """
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    yield BytesIO(obj["Body"].read())


def read_obj_as_buffered_io_xueyuan(bucket, remote_path):
    obj = s3_client.get_object(Bucket=bucket, Key=remote_path)
    body = obj['Body'].read()
    body_io = BytesIO(body)
    return body_io


def read_obj_as_buffered_io_shahe(bucket, remote_path):
    obj = s3_client_shahe.get_object(Bucket=bucket, Key=remote_path)
    body = obj['Body'].read()
    body_io = BytesIO(body)
    return body_io


def list_obj(bucket, directory_path):
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=directory_path
    )

    # 遍历响应中的对象并获取文件名
    file_names = []
    for obj in response.get('Contents', []):
        # 获取文件名，去除目录路径前缀
        file_name = obj['Key'][len(directory_path):]
        file_names.append(file_name)

    return file_names


def put_object_shahe(bucket, body, remote_path):
    s3_client_shahe.put_object(Bucket=bucket, Body=body, Key=remote_path)
