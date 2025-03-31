import torch
import os
import torch.multiprocessing as mp
import torch.distributed as dist


def run_broadcast(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2*rank_id
    print("before broadcast", "Rank", rank_id, " has ", tensor)
    # 从rank 0 广播张量到所有进程
    dist.broadcast(tensor, src=0)
    print("After broadcast", "Rank", rank_id, " has ", tensor)


def run_scatter(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2*rank_id
    print("before scatter", "Rank", rank_id, " has ", tensor)
    if rank_id == 0:
        scatter_list = [torch.tensor([0,0]), torch.tensor([1,1]), torch.tensor([2,2]), torch.tensor([3,3])]
        assert len(scatter_list) == size, "length of scatter_list not equal size"
        print('scater list:', scatter_list)
        dist.scatter(tensor, src=0, scatter_list=scatter_list)
    else:
        dist.scatter(tensor, src=0)
    print("After scatter", "Rank", rank_id, " has ", tensor)


def run_gather(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2*rank_id
    print("Before gather", "Rank", rank_id, " has ", tensor)
    if rank_id == 0:
        gather_list = [torch.zeros(2, dtype=torch.int64) for _ in range(4)]
        print('before gather_list:', gather_list)
        dist.gather(tensor, dst = 0, gather_list=gather_list)
        print('after gather',' Rank ', rank_id, ' has data ', tensor)
        print('gather_list:', gather_list)
    else:
        dist.gather(tensor, dst=0)
        print("After gather", "Rank", rank_id, " has ", tensor)
    

def run_reduce(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2*rank_id
    print("Before reduce", "Rank", rank_id, " has ", tensor)
    dist.reduce(tensor, dst=3, op=dist.ReduceOp.SUM)
    print("After reduce", "Rank", rank_id, " has ", tensor)
 
 
def run_all_gather(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2*rank_id
    gather_list = [torch.zeros(2, dtype=torch.int64) for _ in range(4)]
    print("Before gather", "Rank", rank_id, " has ", tensor)
    dist.all_gather(gather_list, tensor)
    print("After gather", "Rank", rank_id, " has ", tensor)
    print('after gather',' Rank ', rank_id, ' has gather list ', gather_list)


def run_all_reduce(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2*rank_id
    print("Before reduce", "Rank", rank_id, " has ", tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print("After reduce", "Rank", rank_id, " has ", tensor)
        
    
def init_process(rank_id, size, fn, backend='gloo'):
    # 设置分布式环境变量（必须）
    os.environ["MASTER_ADDR"] = '127.0.0.1'  # 主节点IP
    os.environ["MASTER_PORT"] = '29500'  # 通信端口
    dist.init_process_group(backend, rank=rank_id, world_size=size)
    fn(rank_id, size)

if __name__ == '__main__':
    size = 4
    processes = []
    # 设置多进程启动方式为spawn 在win和linux都可以用
    mp.set_start_method('spawn')
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run_reduce))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()            