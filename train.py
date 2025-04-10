from __future__ import print_function, division

import argparse
import os
import random

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config.trainer_config import LOG_BASE_DIR
from data_preprocess.Load_multi_domain_multi_modal import Normaliztion, ToTensor
from data_preprocess.Load_multi_domain_multi_modal import RandomHorizontalFlip
from data_preprocess.Load_multi_domain_multi_modal import Multi_Domain_Spoofing_train
from data_preprocess.Load_multi_domain_multi_modal import Multi_Domain_Spoofing_valtest
from models.model_factory import get_model
from utils.common_util import forward_model, CosineAnnealingLR_with_Restart
from utils.utils_FAS_MultiModal import AvgrageMeter, setup_seed
from utils.utils_FAS_MultiModal import performances_ZeroShot
from accelerate import Accelerator,DeepSpeedPlugin

spoof_templates = [
    'This is an example of a spoof face',
    'This is an example of an attack face',
    'This is not a real face',
    'This is how a spoof face looks like',
    'a photo of a spoof face',
    'a printout shown to be a spoof face',
]

real_templates = [
    'This is an example of a real face',
    'This is a bonafide face',
    'This is a real face',
    'This is how a real face looks like',
    'a photo of a real face',
    'This is not a spoof face',
]
##########    Dataset root    ##########

def ensure_directory_empty(path, clear=False):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"directory {path} created.")
    elif clear:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f'delete failed：{file_path}. {e}')
        print(f"directory {path} cleared.")


def FeatureMap2Heatmap(x, x2, x3, dir_name):
    ## initial images
    org_img = x[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(dir_name + '/x_visual.jpg', org_img)

    org_img = x2[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(dir_name + '/x_depth.jpg', org_img)

    org_img = x3[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(dir_name + '/x_ir.jpg', org_img)


def step_batch(model, sample_batched, optim, accelerator, modality='RGBDIR', mode='train', criterion=None):
    with accelerator.accumulate(model):
        # get the inputs
        spoof_label = sample_batched['spoofing_label']
        inputs = sample_batched['image_x']
        inputs_depth = sample_batched['image_x_depth']
        inputs_ir = sample_batched['image_x_ir']
        domain = sample_batched['domain']



        if mode == 'train':
            logits = forward_model(model, inputs, inputs_depth, inputs_ir,domain, modality)
            # 训练：算loss反向传播，放回结果
            # loss = model.module.cal_loss(spoof_label, criterion)
            loss = model.cal_loss(spoof_label, criterion)
            accelerator.backward(loss["total_loss"])

            optim.step()
            optim.zero_grad()

            return logits, loss
        else:
            logits = forward_model(model, inputs, inputs_depth, inputs_ir,domain, modality)
            # 测试：返回结果
            return logits, None


def train(model, dataloader_train, optim, criterion, epoch, scheduler, accelerator, args):
    model.train()
    metrics = {}
    loop = tqdm(enumerate(dataloader_train), total=len(dataloader_train), position=0,
                disable=not accelerator.is_local_main_process)
    for i, sample_batched in loop:
        _, loss = step_batch(model, sample_batched, optim, accelerator, args.modality, 'train', criterion, )
        n = sample_batched['image_x'].shape[0]
        for key, value in loss.items():
            if key not in metrics.keys():
                metrics[key] = AvgrageMeter()
            metrics[key].update(loss[key].data, n)
        loop.set_description(f'Train Epoch [{epoch}/{args.epochs}]')
        loop.set_postfix(loss=metrics['total_loss'].avg)
    if scheduler is not None and accelerator.is_main_process:
        scheduler.step()


def test(model, dataloader_test, optim, epoch, test_out_filename, save_dir, best_metrics, tensorboardWriter,
         accelerator, args):
    model.eval()

    with torch.no_grad():
        ###########################################
        #          cross-domain    test
        ###########################################
        if accelerator.is_main_process:
            map_score_list = {}

        # 遍历数据集
        for i, sample_batched in enumerate(dataloader_test):
            logits, _ = step_batch(model, sample_batched, optim, accelerator, args.modality, 'test')
            accelerator.wait_for_everyone()
            logits, sample_batched = accelerator.gather_for_metrics([logits, sample_batched])
            if accelerator.is_main_process:
                for key, value in logits.items():
                    if key not in map_score_list:
                        map_score_list[key] = []
                    # 遍历batch里面的图像
                    for test_batch in range(sample_batched['image_x'].shape[0]):
                        map_score = 0.0
                        map_score += F.softmax(logits[key])[test_batch][1]
                        # print(map_score, sample_batched['spoofing_label'])
                        map_score_list[key].append(
                            '{} {}\n'.format(map_score, sample_batched['spoofing_label'][test_batch][0]))

        if accelerator.is_main_process:
            # accelerator.wait_for_everyone()
            # 记录测试结果
            for key, value in map_score_list.items():
                out_name = test_out_filename.replace('main', key)
                with open(out_name, 'w') as file:
                    file.writelines(map_score_list[key])

                ##########################################################################
                #       performance measurement for both intra- and inter-testings
                ##########################################################################
                _, test_AUC, test_HTER = performances_ZeroShot(out_name)
                tensorboardWriter.add_scalar(tag=f'{args.test}/AUC/{key}', scalar_value=test_AUC, global_step=epoch)
                tensorboardWriter.add_scalar(tag=f'{args.test}/HTER/{key}', scalar_value=test_HTER, global_step=epoch)
                # print(f"{key} AUC: {test_AUC}, {key} HTER: {test_HTER}", end=' ')

                index = 0 if key == 'rgb' else (1 if key == 'depth' else 2)
                if key not in best_metrics.keys():
                    best_metrics[key] = {
                        'HTER': 1.0,
                        'AUC': 0.0
                    }
                if test_HTER < best_metrics[key]['HTER']:
                    best_metrics[key]['HTER'] = test_HTER
                    if args.save_best:
                        torch.save(accelerator.unwrap_model(model).state_dict(),
                                   os.path.join(save_dir, f'{key}_best_HTER.pt'))
                    print(f'best {key} HTER: {test_HTER}', end=' ')
                if test_AUC > best_metrics[key]['AUC']:
                    best_metrics[key]['AUC'] = test_AUC
                    if args.save_best:
                        torch.save(accelerator.unwrap_model(model).state_dict(),
                                   os.path.join(save_dir, f'{key}_best_AUC.pt'))
                    print(f'best {key} AUC: {test_AUC}')

        accelerator.wait_for_everyone()


# main function
def train_test():
    args.batchsize = args.batchsize // args.gradient_accumulation
    # 初始化随机种子
    setup_seed(args.seed)
    """加载日志"""
    run_name = f'[{args.tag}]{args.model}_{args.modality}'
    for key, val in vars(args).items():
        if '[model]' in key:
            run_name += f'#{key.replace("[model]", "")}_{val}'
        if 'train' in key or 'test' in key:
            run_name += f'#{key}_{val}'
    log_dir = os.path.join(LOG_BASE_DIR, args.experiment_name)
    # ensure_directory_empty(log_dir)
    tensorboard_dir = os.path.join(log_dir, 'tensorboard', run_name)
    ensure_directory_empty(tensorboard_dir)
    save_dir = os.path.join(log_dir, 'save', run_name)
    ensure_directory_empty(save_dir)
    test_out_filename = os.path.join(save_dir, 'main_out_test.txt')
    tensorboardWriter = SummaryWriter(log_dir=tensorboard_dir)
    print(f'tensorboard command: tensorboard --logdir={tensorboard_dir} --port=6677 --bind_all')
    """data"""
    train_data = Multi_Domain_Spoofing_train(args.train,
                                             transforms.Compose([RandomHorizontalFlip(), ToTensor(), Normaliztion()]))
    dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)
    test_data = Multi_Domain_Spoofing_valtest(args.test, transforms.Compose([Normaliztion(), ToTensor()]))
    dataloader_test = DataLoader(test_data, batch_size=args.batchsize, shuffle=False, num_workers=4)
    """加载模型"""
    model = get_model(args.model, args)
    if args.load != "":
        model.load_state_dict(torch.load(args.load))


    # 确定参数的冻结状态
    # for n, param in model.named_parameters():
    #     print(n, param.requires_grad)


    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.00005)
    # scheduler = CosineAnnealingLR_with_Restart(optim, T_max=50, T_mult=5, eta_min=5e-7)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optim,base_lr=1e-6,max_lr=5e-6)
    scheduler = None
    # deepspeed = DeepSpeedPlugin(zero_stage=2,gradient_clipping=1.0)
    deepspeed = None
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation,deepspeed_plugin = deepspeed)
    model = model.to(accelerator.device)
    model, optim, dataloader_train, dataloader_test, scheduler = accelerator.prepare(
        model, optim, dataloader_train, dataloader_test, scheduler
    )
    criterion = None
    best_metrics = {
        "rgb": {
            'HTER': 1.0,
            'AUC': 0.0
        },
        "depth": {
            'HTER': 1.0,
            'AUC': 0.0
        },
        "ir": {
            'HTER': 1.0,
            'AUC': 0.0
        },
        "mix": {
            'HTER': 1.0,
            'AUC': 0.0
        },

    }

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        ###########################################
        '''                train                '''
        ###########################################
        train(model, dataloader_train, optim, criterion, epoch, scheduler, accelerator, args)
        ###########################################
        '''                test                 '''
        ###########################################
        test(model, dataloader_test, optim, epoch, test_out_filename, save_dir, best_metrics, tensorboardWriter,
             accelerator, args)
    tensorboardWriter.close()
    print('Finished Training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    """
    experiment args
    """
    parser.add_argument('--experiment_name', type=str, default="mm_clip_base", help='大类实验项目名')
    parser.add_argument('--model', type=str, default="MMDA", help='见model_factory.py')
    parser.add_argument('--modality', type=str, default="RGBDIR", help='RGB/D/IR/RGBD/RGBIR/RGBDIR')
    parser.add_argument('--train', nargs='+', default="SURF USC WMCA", help='WMCAGT(WMCA ground test)/')
    parser.add_argument('--test', nargs='+', default="CeFA", help='WMCAGT(WMCA ground test)/')
    parser.add_argument('--tag', default="run_s2", help='run/test/debug')
    """
    train args
    """
    parser.add_argument('--lr', type=float, default=5e-6, help='initial learning rate')  # default=0.0003   0.01
    parser.add_argument('--gradient_accumulation', type=int, default=8, help='gradient_accumulation')  # default=4
    parser.add_argument('--batchsize', type=int, default=24, help='initial batchsize')  # default=16
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--save_best', action='store_true', default=True,
                        help='True  -->  save the best weight; False -->  dont save')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load', type=str, default="", help='weight path')
    """
    model args
    """
    # parser.add_argument('--modal', type=str, default="DEPTH", help='RGB/DEPTH/IR')
    # parser.add_argument('--[model]adapter_dim', type=int, default=8)
    # parser.add_argument('--[model]hidden_size', type=int, default=768, help='ViT的hidden size')

    args = parser.parse_args()

    train_test()

"""
python train.py --lr 5e-6 --tag run --batchsize 16 --gradient_accumulation 4 --modality RGBDIR --model clip --train SURF CeFA USC --test WMCA
python train.py --lr 5e-6 --tag run --batchsize 16 --gradient_accumulation 4 --modality RGBDIR --model clip --train SURF WMCA USC --test CeFA
python train.py --lr 5e-6 --tag run --batchsize 16 --gradient_accumulation 4 --modality RGBDIR --model clip --train WMCA CeFA USC --test SURF
python train.py --lr 5e-6 --tag run --batchsize 16 --gradient_accumulation 4 --modality RGBDIR --model clip --train SURF CeFA WMCA --test USC

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train.py
"""
