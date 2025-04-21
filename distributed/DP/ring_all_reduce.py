import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from rich import print

def ring_all_reduce(tensor, rank, size):
    assert tensor.numel() % size == 0, f"tesnor cannot be equally divided into {size} part, {tensor} has {tensor.numel()}" 
    #chunk_size = tensor.numel() // size
    chunks = tensor.chunk(size)
    buffer = [chunk.clone() for chunk in chunks]
    print("Start Rank", rank, " has tensor", buffer)
    # scatter-reduce
    for step in range(size-1):
        send_idx = (rank+step+1) % size
        recv_idx = (rank+step+2) % size
        # 发送给左边的节点
        next_rank = (rank-1) % size
        send_tensor = buffer[send_idx]
        
        send_req = dist.isend(send_tensor, dst=next_rank)
        # 接收来自右边节点的信息
        before_rank = (rank+1) % size
        recv_tensor = torch.empty_like(buffer[recv_idx])
        recv_req = dist.irecv(recv_tensor, src=before_rank)
        
        send_req.wait()
        recv_req.wait()
        buffer[recv_idx] += recv_tensor
    print("After Reduce, Rank", rank, " has tensor", buffer)
    
    # all_gather 操作
    for step in range(size-1):
        send_idx = (rank+step) % size
        recv_idx = (rank+step+1) % size
        # 发送给左边的节点
        next_rank = (rank-1) % size
        send_tensor = buffer[send_idx]
        send_req = dist.isend(send_tensor, dst=next_rank)
        # 接收来自右边节点的信息
        before_rank = (rank+1) % size
        recv_tensor = torch.empty_like(buffer[recv_idx])
        recv_req = dist.irecv(recv_tensor, src=before_rank)
        
        send_req.wait()
        recv_req.wait()
        buffer[recv_idx] = recv_tensor
    print("Fin Reduce, Rank", rank, " has tensor", buffer)
        
    return torch.cat(buffer)

def init_process(rank_id, size, fn, backend='gloo'):
    # 设置分布式环境变量（必须）
    os.environ["MASTER_ADDR"] = '127.0.0.1'  # 主节点IP
    os.environ["MASTER_PORT"] = '29500'  # 通信端口
    tensor = torch.arange(0, 4, dtype=torch.float32) + (rank_id)*4 + 1
    dist.init_process_group(backend, rank=rank_id, world_size=size)
    fn(tensor, rank_id, size)
       
        
if __name__ == '__main__':
    size = 4
    processes = []
    # 设置多进程启动方式为spawn 在win和linux都可以用
    mp.set_start_method('spawn')
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, ring_all_reduce))
        p.start()
        processes.append(p)
    for p in processes:
        p.join() 
        