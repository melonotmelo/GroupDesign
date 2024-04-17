import math
import os.path
import time

import einops
import torch
from torch import nn
from models.omniarch import OmniArch
from models.revin import RevIN
from models.transformer import Transformer
from others.data_information import *
from random import choice
import test_pb2
import test_pb2_grpc
from concurrent import futures
import grpc
import socket

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def predict_cloud(value1, value2):
    model_ = OmniArch(
        hidden_dim=128,
        patch_size=16,
        num_hidden_layers=12,
        num_t=10,
        train_img_size=64,
        eval_img_size=64,
        channels=4,
        phy_pred_method='step',
        num_heads=8,
        dropout_rate=0.1
    ).to(device)
    import json
    with open('received_data.json', 'r') as file:
        data_middle = json.load(file)
        file.close()
    phy_fls = torch.tensor(data_middle)
    model_.eval()
    with torch.no_grad():  # 不计算梯度，节省计算资源
        pred = model_(phy_fls, value1, value2)
    return pred


class FileTransferService(test_pb2_grpc.FileTransferServiceServicer):
    def SendFile(self, request_iterator, context):
        with open("received_data.json", "wb") as f:
            for chunk in request_iterator:
                f.write(chunk.data)
        return test_pb2.TransferStatus(success=True, message="File received successfully.")

    def EndFileTransfer(self, request, context):
        # 接收结束信号并提取浮点数参数
        value1 = request.value1
        value2 = request.value2
        print(f"Received end signal with values: {value1}, {value2}")
        tensor_data_1 = torch.tensor(value1)
        tensor_data_2 = torch.tensor(value2)
        tensor_shape = torch.Size([1, 1, 1, 1, 4])
        tensor_1 = tensor_data_1.view(tensor_shape)
        tensor_2 = tensor_data_2.view(tensor_shape)
        # 立即使用接收到的值进行推理
        cloud_time_s=time.time()
        inference_result = predict_cloud(tensor_1, tensor_2)
        cloud_time_e=time.time()
        print(f"Predict in Edge: in {cloud_time_e - cloud_time_s} seconds")

        # 返回处理结果
        return test_pb2.EndResponse(success=True, message=f"Inference completed.")


if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    test_pb2_grpc.add_FileTransferServiceServicer_to_server(FileTransferService(), server)
    server.add_insecure_port('10.193.219.84:50051')
    server.start()
    server.wait_for_termination()
