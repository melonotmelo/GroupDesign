import math

import einops
import torch
from torch import nn
import json
from models.revin import RevIN
from models.Jetson import Jetson
import socket
import test_pb2
import test_pb2_grpc
import time
import Pool

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def run_edge():
    model_ = Jetson(
        hidden_dim=128,
        patch_size=16,
        num_hidden_layers=12,
        num_t=10,
        train_img_size=32,
        eval_img_size=32,
        channels=4,
        phy_pred_method='step',
        num_heads=8,
        dropout_rate=0.1
    ).to(device)
    phy_fls_1d = torch.randn((1, 10, 4, 64)).to(device)
    phy_fls_2d = phy_fls_1d.unsqueeze(3).repeat(1, 1, 1, 64, 1)
    model_.eval()
    edge_time_s = time.time()
    with torch.no_grad():  # 不计算梯度，节省计算资源
        pred, mean, stdev = model_(phy_fls_2d)
    # print(mean)
    edge_time_e = time.time()
    print(f"Predict in Edge: in {edge_time_e - edge_time_s} seconds")
    mean_list = mean.flatten().tolist()
    stdev_list = stdev.flatten().tolist()
    # print(mean_list)
    # print(mean.shape)
    pred_list = pred.tolist()
    f2 = open('data_to_send.json', 'w')
    json.dump(pred_list, f2)
    f2.close()

    send_time_s = time.time()
    pool = Pool.GrpcConnectionPool('10.193.219.84:50051')
    channel = pool.get_channel()
    stub = test_pb2_grpc.FileTransferServiceStub(channel)

    def generate_chunks():
        with open("data_to_send.json", "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)  # 读取1MB大小的数据块
                if not chunk:
                    break
                yield test_pb2.FileChunk(data=chunk)

    response = stub.SendFile(generate_chunks())
    print(response.message)
    send_time_e = time.time()
    print(f"Send in Edge: in {send_time_e - send_time_s} seconds")
    end_response = stub.EndFileTransfer(
        test_pb2.EndSignal(message="End of File Transfer", value1=mean_list, value2=stdev_list))
    print(end_response.message)



if __name__ == "__main__":
    run_edge()
