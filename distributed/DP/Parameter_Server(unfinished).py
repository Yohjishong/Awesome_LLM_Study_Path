import torch
import torch.distributed as dist 
import torch.multiprocessing as mp
import os
from rich import print

# 2025.4.3实在不会写了，一直是死锁，学学多进程通信再回来写这个

# 简化一下反向传播过程，就当这个操作是反向传播了hhh
def backward(params):
    return params

# 初始化server节点为2个，hashtable为32，也就是一个server管16个worker
class PS():
    # hashtable = {0:[0,16], 1:[16,32]}
    def __init__(self, rank_ids, params, hash_table):
        self.rank_ids = rank_ids  # dtype set
        self.params = params
        self.hash_table = hash_table
    
    # 添加一个服务节点，主要是分配worker
    def add_server(self, rank_id):
        self.rank_ids.add(rank_id)
        # 找到hash_table中最大的值，拿点空间给新的节点
        max_key = max(self.hash_table.keys(), key=lambda k: self.hash_table[k][1] - self.hash_table[k][0])
        right = self.hash_table[max_key][1]
        left = self.hash_table[max_key][0]
        max_length = right - left
        self.hash_table[max_key][1] = left + max_length//2
        self.hash_table[rank_id][0] = self.hash_table[max_key][1]
        self.hash_table[rank_id][1] = right
    
    # 删除一个服务节点，新添的时候加在右边的空间，所以还回去的时候还左边的空间
    # 如果你要问我初始server被删掉空间怎么办，我只能说我大脑过载了
    def del_server(self, rank_id):
        self.rank_ids.discard(rank_id)
        tar_left = self.hash_table[rank_id][0]
        tar_right = self.hash_table[rank_id][1]
        # 找到左边值等于释放区间右边界的key
        key = [key for key, (left, right) in self.hash_table.items() if right == tar_left]
        self.hash_table[key][1] = tar_right
        
    def server_run(self, rank_id):
        print(f"Before Server {rank_id} has the params {self.params}")
        # 接收worker计算的梯度
        grads = []
        for worker in worker_list:
            grad = torch.zeros_like(self.params)
            dist.recv(grad, src=worker)  # 阻塞接收梯度
            grads.append(grad)
        # 处接收到的梯度
        avg_grad = torch.stack(grads).mean(dim=0)
        self.params -= 0.1 * avg_grad 
        print(f"After Server {rank_id} has the params {self.params}")
        
        left, right = self.hash_table[rank_id]
        
        for worker_id in range(left, right):
            dist.send(self.params, dst=worker_id)
            
    def worker_run(self, worker_id):
        #print(f"Before worker {rank_id} has the params {self.params}")
        # 接收worker计算的梯度
        server_id = [key for key, (start, end) in self.hash_table.items() if start <= worker_id < end]
        params = torch.zeros_like(self.params)
        dist.recv(params, src=server_id)
        grad = backward(params)
        
        dist.send(grad, dst=server_id)


# 初始化
def init_process(rank_id, size,  fn, backend='gloo'):
    os.environ["MASTER_ADDR"] = '127.0.0.1'  # 主节点IP
    os.environ["MASTER_PORT"] = '29500'  # 通信端口
    dist.init_process_group(rank=rank_id, backend=backend, world_size=size)
    params = torch.arange(size) + 1 + rank_id*5
    fn(rank_id, size, params)
    

if __name__ == '__main__':
    # 一共八个节点，其中第一个是server，后面七个都是worker
    size = 5
    server_list = [0]
    worker_list = [1,2,3,4]
    processes = []
    mp.set_start_method('spawn')
    for rank_id in range(size):
        if rank_id == 0:
            # 服务器节点
            p = mp.Process(target=init_process, args=(rank_id, server_list, worker_list, server_run))
        else:
            # 工作节点
            p = mp.Process(target=init_process, args=(rank_id, server_list, worker_list, worker_run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        
        
