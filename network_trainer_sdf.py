import os
import importlib
import json
import logging
from pathlib import Path
import time

from network import lr_schedule
import torch
import torch.nn.functional as F
import tqdm
import yaml
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.profiler import profile, record_function, ProfilerActivity

from dataset import sdf_dataset
from network import criterion, utility
from utils import exp_util

import gc


class TensorboardViz(object):

    def __init__(self, logdir):
        self.logdir = logdir
        self.writter = SummaryWriter(self.logdir)

    def text(self, _text):
        # Enhance line break and convert to code blocks
        _text = _text.replace('\n', '  \n\t')
        self.writter.add_text('Info', _text)

    def update(self, mode, it, eval_dict):
        self.writter.add_scalars(mode, eval_dict, global_step=it)

    def flush(self):
        self.writter.flush()


def load_models(rank, args, device):
    # Load models
    decoder_module = importlib.import_module("network." + args.decoder_name)
    decoder = decoder_module.Model(args.code_length, **args.decoder_specs).to(device)
    encoder_module = importlib.import_module("network." + args.encoder_name)
    encoder = encoder_module.Model(**args.encoder_specs, mode='train').to(device)

    # Load checkpoint
    # if not args.from_scratch or not args.train_encoder:
    #     state_dict = torch.load("ckpt/default/encoder_300.pth.tar")["model_state"]
    #     encoder.load_state_dict(state_dict)
    # if not args.from_scratch or not args.train_decoder:
    #     state_dict = torch.load("ckpt/default/decoder_300.pth.tar")["model_state"]
    #     decoder.load_state_dict(state_dict)

    if args.run_parallel:
        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[rank], output_device=rank)
        decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[rank], output_device=rank)
    
    return encoder, decoder


def prepare_dataloaders(rank, args, train_dataset, val_dataset):
    # Train loader
    if args.run_parallel:
        proc_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=sdf_dataset.collate,
            num_workers=args.num_workers,
            drop_last=True,
            # pin_memory=True,
            sampler=proc_sampler)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=sdf_dataset.collate,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True)
    # Val loader
    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=sdf_dataset.collate,
            num_workers=1,
            drop_last=True,
            pin_memory=True)
    else:
        val_loader = None
    return train_loader, val_loader


def compute_loss(args, loss_func_args, encoder, decoder, batch_loss, batch_stats, data, device, epoch, mode):
    sdf_data = data[0].to(device)  # (B, N, 4)
    surface_data = data[1].to(device)  # (B, N, 6)

    if torch.any(torch.isnan(sdf_data)).item() or torch.any(torch.isnan(surface_data)).item():
        print("nan data")
        return

    '''Encoding'''
    if (not args.train_encoder) or (mode == 'val'):
        torch.set_grad_enabled(False)
    else:
        torch.set_grad_enabled(True)

    lat_vecs = encoder(surface_data)  # (B, L)

    if torch.any(torch.isnan(lat_vecs)).item():
        print(lat_vecs)
        print("lat_vecs is nan!")

    if mode == 'train':
        torch.set_grad_enabled(True)

    # Regulation loss if train encoder
    if args.train_encoder:
        batch_loss.update_loss_dict(criterion.reg_loss(args=loss_func_args, info={"epoch": epoch, "num_epochs": args.num_epochs}, \
                                            latent_vecs=lat_vecs))

    # if mode == 'train' and len(batch_loss) > 0:
    #     batch_loss.get_total_loss().backward(retain_graph=True)

    '''Decoding'''
    sdf_data = sdf_data.reshape(-1, sdf_data.shape[-1])
    num_sdf_samples = sdf_data.shape[0]
    if (args.train_encoder or args.train_decoder):
        batch_split = args.batch_split
        xyz = torch.chunk(sdf_data[:, 0:3], batch_split)
        sdf = torch.chunk(sdf_data[:, 3], batch_split)
        lat_vecs_chunk = lat_vecs.unsqueeze(1).repeat(1, args.num_sdf_samples, 1).view(-1, lat_vecs.size(-1))  # (B_p * S, L)
        lat_vecs_chunk = torch.chunk(lat_vecs_chunk, batch_split)

        for i in range(batch_split):
            if mode == 'train':
                xyz[i].requires_grad_(True)
            net_input = torch.cat([lat_vecs_chunk[i], xyz[i]], dim=1)

            if args.pred_sdf_std:
                sdf_pred, std_pred = decoder(net_input)
                batch_loss.update_loss_dict(criterion.neg_log_likelihood(args=loss_func_args, pd_sdf=sdf_pred, pd_sdf_std=std_pred, \
                                                            gt_sdf=sdf[i], info={"num_sdf_samples": num_sdf_samples}, loss_name='ll'))
            else:
                sdf_pred = decoder(net_input)
                batch_loss.update_loss_dict(criterion.l1_loss(args=loss_func_args, pd_sdf=sdf_pred, \
                                                            gt_sdf=sdf[i], info={"num_sdf_samples": num_sdf_samples}))
            xyz[i].requires_grad_(False)
            if mode == 'train':
                if i < batch_split - 1:
                    batch_loss.get_total_loss().backward(retain_graph=True)
                else:
                    batch_loss.get_total_loss().backward()


def train(rank, args, save_base_dir, train_dataset, val_dataset):
    if args.run_parallel:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda:{}'.format(rank))
    lr_schedules = lr_schedule.get_learning_rate_schedules(args)

    logging.basicConfig(level=logging.INFO)

    # Load models
    encoder, decoder = load_models(rank, args, device)

    # Optimized parameters
    parameters_to_optimize = []
    if args.train_encoder:
        parameters_to_optimize.append({"params": encoder.parameters(), "lr": lr_schedules[0].get_learning_rate(1)})
    else:
        encoder.eval()
    if args.train_decoder:
        parameters_to_optimize.append({"params": decoder.parameters(), "lr": lr_schedules[1].get_learning_rate(0)})
    else:
        decoder.eval()

    optimizer_all = torch.optim.Adam(parameters_to_optimize)

    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloaders(rank, args, train_dataset, val_dataset)
    
    # Loss funtions args
    loss_func_args = exp_util.dict_to_args(args.training_loss)
    
    # Epoch bar, tensorboard
    start_epoch = 1
    if rank == 0:
        epoch_bar = tqdm.trange(start_epoch, args.num_epochs + 1, desc='epochs')
        logging.info("starting from epoch {}".format(start_epoch))
        logging.info("Number of decoder parameters: {}".format(sum(p.data.nelement() for p in decoder.parameters())))
        logging.info("Number of encoder parameters: {}".format(sum(p.data.nelement() for p in encoder.parameters())))

        viz = TensorboardViz(logdir=str(save_base_dir / 'tensorboard'))
        viz.text(yaml.dump(vars(args)))
    else:
        epoch_bar = list(range(start_epoch, args.num_epochs + 1))       

    if args.run_parallel:
        torch.distributed.barrier() 

    train_it = 0
    val_it = 0
    # Epoch loop for train / val
    for epoch in range(1, args.num_epochs + 1):
        # Update learning rate
        lr_schedule.adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        # Only record results in the main thread
        if rank == 0:
            train_meter = exp_util.AverageMeter()
            train_running_meter = exp_util.RunningAverageMeter(alpha=0.3)
            val_meter = exp_util.AverageMeter()
            val_running_meter = exp_util.RunningAverageMeter(alpha=0.3)
            stats_train_meter = exp_util.AverageMeter()
            stats_val_meter = exp_util.AverageMeter()
            batch_bar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train')

        # Training
        if args.train_encoder:
            encoder.train()
        if args.train_decoder:
            decoder.train()
        
        for data_list, idx in train_loader:
            # print(data_list)
            if len(data_list) == 0:
                continue
                
            batch_loss = exp_util.CombinedChunkLoss()
            batch_stats = exp_util.CombinedChunkLoss()
            optimizer_all.zero_grad() 

            compute_loss(args, loss_func_args, encoder, decoder, batch_loss, batch_stats, data_list, device, epoch, 'train')
            optimizer_all.step() 

            loss_res = batch_loss.get_accumulated_loss_dict()
            stats_res = batch_stats.get_accumulated_loss_dict()
            del batch_loss, batch_stats
            
            train_it += 1     
            if rank == 0:
                batch_bar.update()
                train_running_meter.append_loss(loss_res)
                train_meter.append_loss(loss_res)
                stats_train_meter.append_loss(stats_res)
                # logging.info('train iter: {}, loss: {}, stats: {}'.format(train_it, dict(loss_res), dict(stats_res)))
                batch_bar.set_postfix(train_running_meter.get_loss_dict())
                if train_it % 50 == 0:
                    for loss_name, loss_val in loss_res.items():
                        viz.update('train/' + loss_name, train_it, {'scalar': loss_val})
                    for stats_name, stats_val in stats_res.items():
                        viz.update('train/' + stats_name, val_it, {'scalar': stats_val})

            if args.run_parallel:
                torch.distributed.barrier()     

        if rank == 0:
            batch_bar.close()

        # Validation only in the main thread
        if rank == 0:
            if args.train_encoder:
                encoder.eval()
            if args.train_decoder:
                decoder.eval()

            batch_bar = tqdm.tqdm(total=len(val_loader), leave=False, desc='val')
            for data_list, idx in val_loader:
                if len(data_list) == 0:
                    continue
                batch_loss = exp_util.CombinedChunkLoss()
                batch_stats = exp_util.CombinedChunkLoss()
                compute_loss(args, loss_func_args, encoder, decoder, batch_loss, batch_stats, data_list, device, epoch, 'val')

                loss_res = batch_loss.get_accumulated_loss_dict()
                stats_res = batch_stats.get_accumulated_loss_dict()
                # logging.info('val iter: {}, loss: {}, stats: {}'.format(val_it, dict(loss_res), dict(stats_res)))
                del batch_loss, batch_stats

                val_it += 1 
                batch_bar.update()    
                val_running_meter.append_loss(loss_res)
                val_meter.append_loss(loss_res)
                stats_val_meter.append_loss(stats_res)
                batch_bar.set_postfix(val_running_meter.get_loss_dict())
                if val_it % 50 == 0:
                    for loss_name, loss_val in loss_res.items():
                        viz.update('val/' + loss_name, val_it, {'scalar': loss_val})
                    for stats_name, stats_val in stats_res.items():
                        viz.update('val/' + stats_name, val_it, {'scalar': stats_val})
            
            batch_bar.close()

        # Record epoch statistics & store checkpoints
        if rank == 0:
            train_avg = train_meter.get_mean_loss_dict()
            val_avg = val_meter.get_mean_loss_dict()
            stats_train_avg = stats_train_meter.get_mean_loss_dict()
            stats_val_avg = stats_val_meter.get_mean_loss_dict()
            logging.info("#################################################################################################")
            logging.info('Epoch: {}, Train loss: {}, Stats: {}, Total: {}'.format(epoch, train_avg, stats_train_avg, sum(train_avg.values())))
            logging.info('Epoch: {}, Valid loss: {}, Stats: {}, Total: {}'.format(epoch, val_avg, stats_val_avg, sum(val_avg.values())))
            logging.info("#################################################################################################")
            for meter_key, meter_val in train_avg.items():
                viz.update("epoch_sum/" + meter_key, epoch, {"train": meter_val})
            viz.update("epoch_sum/loss", epoch, {"train": sum(train_avg.values())})
            for meter_key, meter_val in stats_train_avg.items():
                viz.update("epoch_sum/" + meter_key, epoch, {"train": meter_val})
            for meter_key, meter_val in val_avg.items():
                viz.update("epoch_sum/" + meter_key, epoch, {"val": meter_val})
            viz.update("epoch_sum/loss", epoch, {"val": sum(val_avg.values())})
            for meter_key, meter_val in stats_val_avg.items():
                viz.update("epoch_sum/" + meter_key, epoch, {"val": meter_val})
            for sid, schedule in enumerate(lr_schedules):
                viz.update(f"train_stat/lr_{sid}", epoch, {'scalar': schedule.get_learning_rate(epoch)})
            
            if epoch in args.checkpoints:
                if args.run_parallel:
                    encoder_model = encoder.module.state_dict()
                    decoder_model = decoder.module.state_dict()
                else:
                    encoder_model = encoder.state_dict()
                    decoder_model = decoder.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state": encoder_model,
                }, save_base_dir / f"encoder_{epoch}.pth.tar")
                torch.save({
                    "epoch": epoch,
                    "optimizer_state": optimizer_all.state_dict(),
                    # "latent_vec": all_lat_vecs
                }, save_base_dir / f"training_{epoch}.pth.tar")
                torch.save({
                    "epoch": epoch,
                    "model_state": decoder_model,
                }, save_base_dir / f"decoder_{epoch}.pth.tar")    
        
        if args.run_parallel:
            torch.distributed.barrier()           


def main():
    parser = exp_util.ArgumentParserX(add_hyper_arg=True)
    args = parser.parse_args()

    # Latent code length
    args.encoder_specs.update({"latent_size": args.code_length})
    # Predict uncertainty of sdf or not
    args.decoder_specs.update({"pred_std": args.pred_sdf_std})
    # Checkpoints
    checkpoints = list(range(args.snapshot_frequency, args.num_epochs + 1, args.snapshot_frequency))
    for checkpoint in args.additional_snapshots:
        checkpoints.append(checkpoint)
    checkpoints.sort()
    args.checkpoints = checkpoints

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    # Num of GPU
    if args.run_parallel:
        logging.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
        args.world_size = torch.cuda.device_count()
        if args.world_size == 1:
            args.run_parallel = False
            logging.info("Disable parallel training!")
    else:
        logging.info("Let's use 1 GPU!")
    
    print(args.code_length)
    print(args.num_epochs)
    print(args.num_workers)
    print("Train encoder: ", args.train_encoder)
    print("Train decoder: ", args.train_decoder)

    # Save arguments
    save_base_dir = Path("di-checkpoints/%s" % args.run_name)
    assert not save_base_dir.exists()
    save_base_dir.mkdir(parents=True, exist_ok=True)

    with (save_base_dir / "hyper.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    # Train / val split
    dataset = sdf_dataset.SDFCombinedDataset(*[
        sdf_dataset.SDFDataset(**t, num_sample=args.num_sdf_samples) for t in args.train_set])    

    train_size = min(int(0.9 * len(dataset)), len(dataset) - 1)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    logging.info("Train dataset size: {}".format(train_size))
    logging.info("Val dataset size: {}".format(val_size))

    if args.run_parallel:
        # Launch parallel processes
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(train, nprocs=args.world_size, args=(args, save_base_dir, train_dataset, val_dataset))
    else:
        train(0, args, save_base_dir, train_dataset, val_dataset)



if __name__ == '__main__':
    main()
