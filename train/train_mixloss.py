from __future__ import division
import warnings
from Networks.models import base_patch16_384_token, base_patch16_384_gap, base_patch16_384_attention, \
    base_patch16_384_fgap, base_patch16_384_swin, base_patch16_384_effnet, base_patch16_384_mamba, \
    base_patch16_384_swin1, base_patch16_384_swin_llm, base_patch16_384_swin_dk, base_patch16_384_swin_multi
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import save_checkpoint, setup_seed
import torch
import os
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import numpy as np
from image import load_data

# from torch.utils.data.distributed import DistributedSampler
warnings.filterwarnings('ignore')
import time

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')


def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy'
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
        # print("------",train_lists)
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    print(len(train_list), len(val_list))

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']  #####

    if args['model_type'] == 'token':
        model = base_patch16_384_token(pretrained=True)
    elif args['model_type'] == 'gap':
        model = base_patch16_384_gap(pretrained=True)
    elif args['model_type'] == 'attention':
        model = base_patch16_384_attention(pretrained=True)
    elif args['model_type'] == 'swin':
        model = base_patch16_384_swin(pretrained=True, mode=args['mode'])

    elif args['model_type'] == 'swin1':
        model = base_patch16_384_swin1(pretrained=True, mode=args['mode'])

    elif args['model_type'] == 'swinllm':
        model = base_patch16_384_swin_llm(pretrained=True, mode=args['mode'])

    elif args['model_type'] == 'swindk':
        model = base_patch16_384_swin_dk(pretrained=True, mode=args['mode'])

    elif args['model_type'] == 'swinmulti':
        model = base_patch16_384_swin_multi(pretrained=True, mode=args['mode'])

    elif args['model_type'] == 'fgap':
        model = base_patch16_384_fgap(pretrained=True)
    elif args['model_type'] == 'effnet':
        model = base_patch16_384_effnet()
    elif args['model_type'] == 'mamba':
        model = base_patch16_384_mamba(pretrained=True, mode=args['mode'])
    else:
        print("Do not have the network: {}".format(args['model_type']))
        exit(0)

    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))

    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    #  、####
    # if args.local_rank != -1:
    #    torch.cuda.set_device(args.local_rank)
    #    device=torch.device("cuda", args.local_rank)
    #    torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # model.to(device)
    # num_gpus = torch.cuda.device_count()
    # if num_gpus > 1:
    #     print('use {} gpus!'.format(num_gpus))
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #                                             output_device=args.local_rank)
    ########

    # criterion = nn.L1Loss(size_average=False).cuda()

    # criterion = MAPELoss(eps=1e-6).cuda()  # 原MAPE损失
    # criterion = MixedLoss(eps=1e-3, alpha=0.5).cuda()  # 混合损失

    # criterion = MAPELoss(eps=1e-6).cuda()  # 原MAPE损失

    # criterion = StableMAPELoss(eps=1e-2, threshold=10).cuda()  # 稳定版MAPE损失

    # 小人群密集场景
    criterion = EnhancedStableMAPELoss(
        eps=5e-3, threshold=5, small_weight=0.8, alpha=0.3,
        dynamic_threshold=True, percentile=25, use_quantile=True, q=0.6
    ).cuda()

    # # 大人群稀疏场景
    # criterion = EnhancedStableMAPELoss(
    #     eps=1e-2, threshold=15, small_weight=0.6, alpha=0.4,
    #     dynamic_threshold=False, use_quantile=False
    # ).cuda()

    # 平衡大小目标
    # criterion = EnhancedStableMAPELoss(
    #     eps=1e-2, threshold=10, small_weight=0.7, alpha=0.3,
    #     dynamic_threshold=True, percentile=30, use_quantile=True, q=0.5
    # ).cuda()

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=-1)

    print(args['pre'])

    # args['save_path'] = args['save_path'] + str(args['rdt'])
    print(args['save_path'])
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])

    print(args['best_pred'], args['start_epoch'])
    train_data = pre_data(train_list, args, train=True)
    test_data = pre_data(val_list, args, train=False)

    for epoch in range(args['start_epoch'], args['epochs']):

        start = time.time()
        train(train_data, model, criterion, optimizer, epoch, args, scheduler)
        end1 = time.time()

        if epoch % 1 == 0 and epoch >= 10:
            prec1 = validate(test_data, model, args)
            end2 = time.time()
            is_best = prec1 < args['best_pred']
            args['best_pred'] = min(prec1, args['best_pred'])

            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, is_best, args['save_path'])


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

        '''for debug'''
        # if j> 10:
        #     break
    return data_keys

def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()

    for i, (fname, img, gt_count) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()

        output = model(img)
        # 检查 output 是否为元组
        if isinstance(output, tuple):
            # 假设第一个元素是计数输出
            out1 = output[0]
        else:
            out1 = output

        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)

        loss = criterion(out1, gt_count)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        # if i % 10 == 0:
        #     with torch.no_grad():
        #         pred = model(img.cuda())
        #         if isinstance(pred, tuple):
        #             pred = pred[0]
        #         pred_np = pred.cpu().numpy().flatten()  # 已正确使用.cpu()
        #         gt_np = gt_count.cpu().numpy().flatten()  # 新增.cpu()
        #
        #     print(f"Epoch {epoch}, Batch {i}:")
        #     print(f"  预测值: 最小值={pred_np.min():.2f}, 最大值={pred_np.max():.2f}, 均值={pred_np.mean():.2f}")
        #     print(f"  真实值: 最小值={gt_np.min():.2f}, 最大值={gt_np.max():.2f}, 均值={gt_np.mean():.2f}")
        #     print(f"  MAE={np.mean(np.abs(pred_np - gt_np)):.2f}")

    scheduler.step()


def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    for i, (fname, img, gt_count) in enumerate(test_loader):

        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            out1 = model(img)
            count = torch.sum(out1).item()

        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        # if i % 15 == 0:
        #     print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# class MAPELoss(nn.Module):
#     def __init__(self, eps=1e-3):  # 增大eps至1e-3，避免分母过小
#         super(MAPELoss, self).__init__()
#         self.eps = eps
#
#     def forward(self, pred, target):
#         pred = pred.float()
#         target = target.float()
#
#         # 计算绝对误差
#         abs_diff = torch.abs(pred - target)
#
#         # 改进分母：添加平滑项并使用|target|+eps
#         denom = torch.abs(target) + self.eps
#
#         # 计算MAPE并平均
#         mape = abs_diff / denom
#         loss = 100.0 * torch.mean(mape)
#
#         return loss
#
#
# class MixedLoss(nn.Module):
#     # def __init__(self, eps=1e-3, alpha=0.5):  # alpha控制MAPE和MAE的权重
#     def __init__(self, eps=1e-3, alpha=0.3):  # alpha控制MAPE和MAE的权重
#         super(MixedLoss, self).__init__()
#         self.eps = eps
#         self.alpha = alpha
#         self.mae = nn.L1Loss()
#
#     def forward(self, pred, target):
#         pred = pred.float()
#         target = target.float()
#
#         # MAPE部分
#         abs_diff = torch.abs(pred - target)
#         denom = torch.abs(target) + self.eps
#         mape = abs_diff / denom
#         mape_loss = 100.0 * torch.mean(mape)
#
#         # MAE部分
#         mae_loss = self.mae(pred, target)
#
#         # 混合损失
#         loss = self.alpha * mape_loss + (1 - self.alpha) * mae_loss
#         return loss


class StableMAPELoss(nn.Module):
    def __init__(self, eps=1e-2, threshold=10):
        super(StableMAPELoss, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        # 调试信息：打印预测值和真实值的统计
        if self.training and torch.rand(1) < 0.01:  # 1%概率打印，避免频繁输出
            print(
                f"Prediction min: {pred.min().item():.4f}, max: {pred.max().item():.4f}, mean: {pred.mean().item():.4f}")
            print(
                f"Target min: {target.min().item():.4f}, max: {target.max().item():.4f}, mean: {target.mean().item():.4f}")

        # 计算绝对误差
        abs_diff = torch.abs(pred - target)

        # 识别小目标
        small_mask = (torch.abs(target) < self.threshold).float()

        # 对小目标使用MAE
        small_loss = self.mae(pred[small_mask > 0.5], target[small_mask > 0.5]) if torch.sum(small_mask) > 0 else 0

        # 对大目标使用稳定的MAPE
        large_mask = 1 - small_mask
        if torch.sum(large_mask) > 0:
            denom = torch.abs(target[large_mask > 0.5]) + self.eps
            large_mape = abs_diff[large_mask > 0.5] / denom
            large_loss = 100.0 * torch.mean(large_mape)
        else:
            large_loss = 0

        # 组合损失
        total_loss = (torch.sum(small_mask) * small_loss + torch.sum(large_mask) * large_loss) / (
                    torch.sum(small_mask) + torch.sum(large_mask) + 1e-8)

        # 打印MAPE值供调试
        if self.training and torch.rand(1) < 0.01:
            print(f"Batch MAPE: {total_loss.item():.2f}%")

        return total_loss


class EnhancedStableMAPELoss(nn.Module):
    def __init__(self,
                 eps=1e-2,  # 避免分母为零的平滑项
                 threshold=10,  # 区分小目标和大目标的阈值
                 small_weight=0.7,  # 小目标损失的权重
                 alpha=0.5,  # MAPE与MAE的平衡系数
                 dynamic_threshold=False,  # 是否使用动态阈值
                 percentile=25,  # 动态阈值的百分位数
                 use_quantile=False,  # 是否使用分位数损失
                 q=0.5):  # 分位数参数
        super(EnhancedStableMAPELoss, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.small_weight = small_weight
        self.alpha = alpha
        self.dynamic_threshold = dynamic_threshold
        self.percentile = percentile
        self.use_quantile = use_quantile
        self.q = q
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        # 调试信息：打印预测值和真实值的统计
        if self.training and torch.rand(1) < 0.01:
            print(
                f"Prediction min: {pred.min().item():.4f}, max: {pred.max().item():.4f}, mean: {pred.mean().item():.4f}")
            print(
                f"Target min: {target.min().item():.4f}, max: {target.max().item():.4f}, mean: {target.mean().item():.4f}")

        # 动态阈值计算
        if self.dynamic_threshold:
            batch_threshold = torch.quantile(torch.abs(target), self.percentile / 100.0).item()
        else:
            batch_threshold = self.threshold

        # 计算绝对误差
        abs_diff = torch.abs(pred - target)

        # 识别小目标
        small_mask = (torch.abs(target) < batch_threshold).float()

        # 对小目标使用MAE或分位数损失
        if torch.sum(small_mask) > 0:
            small_pred = pred[small_mask > 0.5]
            small_target = target[small_mask > 0.5]

            if self.use_quantile:
                # 分位数损失对异常值更鲁棒
                error = small_target - small_pred
                small_loss = torch.mean(torch.max(self.q * error, (self.q - 1) * error))
            else:
                small_loss = self.mae(small_pred, small_target)
        else:
            small_loss = 0

        # 对大目标使用稳定的MAPE
        large_mask = 1 - small_mask
        if torch.sum(large_mask) > 0:
            # 改进的分母：使用目标值的绝对值 + eps
            denom = torch.abs(target[large_mask > 0.5]) + self.eps
            large_mape = abs_diff[large_mask > 0.5] / denom
            large_loss = 100.0 * torch.mean(large_mape)
        else:
            large_loss = 0

        # 使用权重组合大小目标损失
        mapeloss = self.small_weight * small_loss + (1 - self.small_weight) * large_loss

        # 计算纯MAE和MSE
        total_mae = self.mae(pred, target)
        total_mse = self.mse(pred, target)

        # 混合损失：平衡MAPE和MAE
        final_loss = self.alpha * mapeloss + (1 - self.alpha) * total_mae

        # 打印损失值供调试
        if self.training and torch.rand(1) < 0.01:
            print(f"Batch MAPE: {mapeloss.item():.2f}%, MAE: {total_mae.item():.2f}, MSE: {total_mse.item():.2f}")
            print(
                f"Threshold: {batch_threshold:.2f}, Small targets: {torch.sum(small_mask).item()}, Large targets: {torch.sum(large_mask).item()}")

        return final_loss

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)