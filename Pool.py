import grpc
import test_pb2_grpc

class GrpcConnectionPool:
    def __init__(self, target, pool_size=5):
        self._target = target
        self._pool_size = pool_size
        self._pool = [grpc.insecure_channel(target) for _ in range(pool_size)]
        self._current = 0

    def get_channel(self):
        channel = self._pool[self._current]
        self._current = (self._current + 1) % self._pool_size
        return channel

# 使用连接池
pool = GrpcConnectionPool('10.193.219.84:50051')
channel = pool.get_channel()
stub = test_pb2_grpc.FileTransferServiceStub(channel)