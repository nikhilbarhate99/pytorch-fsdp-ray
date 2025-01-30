
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# Initialize Ray.
import ray


import os
import time
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from typing import Optional, Callable

from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import wandb


def get_gpu_usage(device="cuda:0"):
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) // 1024 ** 2
    return mem_used_MB



def track_gpu_memory(description: str = "default") -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get description (function name if not provided)
            desc = description or func.__name__
            before_mem = get_gpu_usage()
            # Execute the function
            result = func(*args, **kwargs)
            after_mem = get_gpu_usage()

            print(f"[{desc}] before: {before_mem:.2f} MB | after: {after_mem:.2f} MB | change: {after_mem - before_mem:.2f} MB")
            
            return result
        return wrapper
    return decorator


def setup_wandb(rank, name, project, config):
    if rank == 0:
        run = wandb.init(
            name=name,
            project=project,
            config=config,
        )


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.fc = nn.Sequential(
            nn.Linear(9216, 16384),
            nn.ReLU(),
            torch.nn.Dropout(0.5),
            nn.Linear(16384, 16384),
            nn.ReLU(),
            torch.nn.Dropout(0.5),
            nn.Linear(16384, 128),
            nn.ReLU(),
            torch.nn.Dropout(0.5),
            nn.Linear(128, 10)
        )


    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        train_loss = round((ddp_loss[0] / ddp_loss[1]).item(), 6)
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, train_loss))

        wandb.log({"train/loss": train_loss}, step=epoch)


def test(model, rank, world_size, test_loader, epoch):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_total_count = int(ddp_loss[2])
        test_correct_count = int(ddp_loss[1])
        test_acc = 100. * test_correct_count/ test_total_count

        test_loss = round((ddp_loss[0] / test_total_count).item(), 6)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, test_correct_count, test_total_count, test_acc))
        

        wandb.log({"test/loss": test_loss}, step=epoch)
        wandb.log({"test/acc": test_acc}, step=epoch)



def _move_to_device_helper(obj, device):
    if torch.is_tensor(obj):
        print("moving optim wts")
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: _move_to_device_helper(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_move_to_device_helper(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_move_to_device_helper(item, device) for item in obj)
    return obj


@track_gpu_memory("moving optimizer to device")
def move_optimizer_to_device(optimizer, device):
    # _move_to_device_helper(optimizer.state_dict()["state"], device)
    for param_id, state in optimizer.state_dict()["state"].items():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)
    dist.barrier()
    torch.cuda.empty_cache()


@track_gpu_memory("moving model to device")
def move_model_to_device(model, device):
    model.to(device=device)        
    dist.barrier()
    torch.cuda.empty_cache()


@track_gpu_memory("clearing torch cuda cache")
def clear_torch_cuda_cache():
    dist.barrier()
    torch.cuda.empty_cache()


@ray.remote
def fsdp_main(rank, world_size, args):

    print("-" * 100)
    print(f"begin of fsdp main gpu mem: {get_gpu_usage()} MB")
    
    pin_memory = False

    setup(rank, world_size)

    setup_wandb(
        rank=rank, 
        name="ray_fsdp_v1",
        project="pytorch-fsdp-ray",
        config={}
    )


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    test_dataset = datasets.MNIST('../data', train=False,
                        transform=transform)


    og_len = len(train_dataset)
    train_dataset = Subset(train_dataset, [i for i in range(args.num_train_data)])
    print(f"trimming training dataset_len from {og_len} to {len(train_dataset)}")

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}

    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': pin_memory,
                    'shuffle': False}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = ConvNet().to(rank)

    model = FSDP(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()

    print("-" * 100)
    print(f"begin of training gpu mem: {get_gpu_usage()} MB")
    for epoch in range(0, args.epochs):
        print("-" * 100)
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        print(f"after train, gpu mem: {get_gpu_usage()} MB")
        print("-" * 30)
        
        
        ### chek for dtype issues (fp16 to fp32) ????
        
        clear_torch_cuda_cache()
        move_model_to_device(model, "cpu")
        move_optimizer_to_device(optimizer, "cpu")
        
        print("-" * 30)
        time.sleep(0.5)
        
        clear_torch_cuda_cache()
        move_model_to_device(model, rank)
        move_optimizer_to_device(optimizer, rank)

        print("-" * 30)
        test(model, rank, world_size, test_loader, epoch)
        scheduler.step()
        print(f"after test, gpu mem: {get_gpu_usage()} MB")

    print("-" * 100)


    init_end_event.record()

    if rank == 0:
        init_end_event.synchronize()
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()


"""
Refer to vLLM Ray discussion https://github.com/vllm-project/vllm/discussions/691#discussioncomment-10730861
"""

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--num-train-data', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)


    print("=" * 100)
    print("ray init")
    ray.init()

    print("=" * 100)
    print("running fsdp with ray!")

    WORLD_SIZE = torch.cuda.device_count()
    NUM_CPU_PER_WORKER = 1
    NUM_GPU_PER_WORKER = 1
    
    # mp.spawn(fsdp_main,
    #     args=(WORLD_SIZE, args),
    #     nprocs=WORLD_SIZE,
    #     join=True) 


    pg = placement_group(
        name="ray_fsdp_pg",
        bundles=[{"CPU": NUM_CPU_PER_WORKER, "GPU": NUM_GPU_PER_WORKER} for _ in range(WORLD_SIZE)],
        strategy="STRICT_PACK"
    )

    ray.get(pg.ready(), timeout=10)
    print(placement_group_table(pg))

    futures = []
    for i in range(WORLD_SIZE):
        futures.append(
            fsdp_main.options(
                num_cpus=NUM_CPU_PER_WORKER, 
                num_gpus=NUM_GPU_PER_WORKER,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote(rank=i, world_size=WORLD_SIZE, args=args)
        )

    results = [ray.get(f) for f in futures]


    print("=" * 100)
    print("ray shutdown")
    ray.shutdown()
    print("=" * 100)



