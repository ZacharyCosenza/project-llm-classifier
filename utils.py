import os
import torch

def create_logger(log_path, rank=0):
    def log(msg, rank_specific=False):
        if rank_specific or rank == 0:
            print(msg)
            if rank == 0:
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
    return log

def setup_device_and_distributed():
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if "LOCAL_RANK" in os.environ:
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend="gloo")
        use_ddp, world_size, rank = True, torch.distributed.get_world_size(), torch.distributed.get_rank()
    elif num_gpus >= 1:
        device, use_ddp, world_size, rank = torch.device("cuda"), False, 1, 0
    else:
        device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
        use_ddp, world_size, rank = False, 1, 0
    return device, use_ddp, world_size, rank
