from utils import h36motion3d as datasets
from model import AttModel
from utils import CMU_motion_3d as CMU_Motion3D
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim
from model import AttModel_CMU


def main(opt):
    in_features = opt.in_features  # 75
    d_model = opt.d_model  # 256
    kernel_size = opt.kernel_size  # 10
    # 实例化模型
    # kernel_size = 10, num_stage = 12, dct_n = 20
    print('>>> create models')
    net_pred = AttModel_CMU.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                     num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.cuda()

    model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    # start_epoch = ckpt['epoch'] + 1
    # err_best = ckpt['err']
    # lr_now = ckpt['lr']

    # 保存的模型参数加载进入
    net_pred.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    head = np.array(['act'])
    for k in range(1, opt.output_n + 1):
        head = np.append(head, [f'#{k}'])

    acts = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
            "washwindow"]

    errs = np.zeros([len(acts) + 1, opt.output_n])
    test_loader = {}
    for i, act in enumerate(acts):

        # i从0到14,遍历所有动作类别数据集
        test_dataset = CMU_Motion3D.CMU_Motion3D(opt=opt, split=2, actions=act)
        print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
        test_loader[act] = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True)
        time_st = time.time()

        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader[act], opt=opt)
        time_end = time.time()
        time_takes = time_end - time_st
        print(time_takes)
        print('testing error: {:.3f}'.format(ret_test['#1']))
        # ['#值为要预测的帧数']

        # log文件记录结果
        ret_log = np.array([])

        # 将每个动作中的第1到第25帧的预测结果保留到一行数组，ret_test[k]代表第k帧的预测值
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])

        # 将每行(i)结果ret_log写入errs数组中
        errs[i] = ret_log

    errs[-1] = np.mean(errs[:-1], axis=0)
    # 最后一行计算所有动作的平均值，按列（每一帧）计算

    acts = np.expand_dims(np.array(acts + ["average"]), axis=1)
    # 扩展最后一行

    value = np.concatenate([acts, errs.astype(str)], axis=1)
    # 将动作标题与预测数据合并

    log.save_csv_log(opt, head, value, is_create=True, file_name='test_pre_action')


def run_model(net_pred, is_train=0, data_loader=None, opt=None):
    net_pred.eval()
    # titles 代表预测的帧数， titles = [1, 2, ..., 25]
    titles = np.array(range(opt.output_n)) + 1
    m_p3d_h36 = np.zeros([opt.output_n])
    # 累积batch_size，方便计算平均误差
    n = 0
    # in_n = 50
    in_n = opt.input_n
    # out_n = 25
    out_n = opt.output_n
    dim_used = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32,
                         33, 34, 35, 36, 37, 38, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56,
                         57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 80,
                         84, 85, 86, 90, 91, 92, 93, 94, 95, 96, 97, 98, 102, 103, 104, 105, 106, 107,
                         111, 112, 113])
    # kernel_size = 10，seq_in代表 M + T 中的 M ，
    seq_in = opt.kernel_size
    # 去掉不要的关节和相应的维度
    joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
    # 一次预测10帧，迭代3次预测25帧
    itera = 3
    # idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
    #         out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))

    # p3d_h36 维度为tensor(32, 75, 96), batch_size是32，取的一个 5(50+25) 是一个 ground_truth 样本大小
    for i, (p3d_h36) in enumerate(data_loader):

        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size

        p3d_h36 = p3d_h36.float().cuda()
        p3d_src = p3d_h36.clone()[:, :, dim_used]

        # input_n = 50, output_n = 10 , itera = 3
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=10, itera=itera)

        p3d_out_all = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
        p3d_out = p3d_out.reshape([-1, out_n, 38, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 38, 3])

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
    ret = {}
    m_p3d_h36 = m_p3d_h36 / n
    for j in range(out_n):
        ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
