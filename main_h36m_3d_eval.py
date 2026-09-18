from utils import h36motion3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim


def main(opt):
    # opt.is_eval = True
    print('>>> create models')
    in_features = 66
    d_model = opt.d_model
    # d_model 256
    kernel_size = opt.kernel_size
    # 实例化模型
    # kernel_size = 10, num_stage = 12, dct_n = 20
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    model_path_len = '{}/ckpt_last.pth.tar'.format(opt.ckpt)
    print(">>> loading ckpt len firom '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    # start_epoch = ckpt['epoch'] + 1
    # err_best = ckpt['err']
    # lr_now = ckpt['lr']

    # 保存的模型参数加载进入
    net_pred.load_state_dict(ckpt['state_dict'])
    # # ----------------------------------------------------------------------------------------------------------
    # layer = dict(net_pred.named_parameters())
    # weight = layer['gcn.gcbs.11.gc2.att']
    # print(f"Layer: {'gcn.gcbs.11.gc2.att'} Weights:\n")
    # attmatrix = weight.data
    # print(attmatrix.shape)
    # matrix = attmatrix.cpu().numpy()
    # plt.imshow(matrix, cmap='viridis', interpolation='none')
    # plt.colorbar()
    # plt.title('Learned graph adjacency matrix')
    # plt.show()
    #
    # # 从邻接矩阵创建图
    # G = nx.from_numpy_matrix(matrix)
    #
    # # 绘制图
    # pos = nx.spring_layout(G)  # 选择布局
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=16)
    # plt.title('Graph Visualization')
    # plt.show()
    # # ------------------------------------------------------------------------------------------------------------
    total_params = sum(p.numel() for p in net_pred.parameters())
    print(f'Total number of parameters: {total_params}')

    print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    head = np.array(['act'])
    for k in range(1, opt.output_n + 1):
        head = np.append(head, [f'#{k}'])

    # acts = ["walking", "eating", "smoking", "discussion", "directions",
    #         "greeting", "phoning", "posing", "purchases", "sitting",
    #         "sittingdown", "takingphoto", "waiting", "walkingdog",
    #         "walkingtogether"]
    acts = ["waiting"]
    errs = np.zeros([len(acts) + 1, opt.output_n])
    print(net_pred)
    # tagert_layer = net_pred.get_layer('GC_Block')
    all = 0
    for i, act in enumerate(acts):

        # i从0到14,遍历所有动作类别数据集
        test_dataset = datasets.Datasets(opt, split=2, actions=[act])
        print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
        test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True)
        time_st = time.time()

        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        time_end = time.time()
        time_takes = time_end - time_st
        print(time_takes)
        print('testing error: {:.3f}'.format(ret_test['#1']))
        # ['#值为要预测的帧数']

        # log文件记录结果
        ret_log = np.array([])

        # 将第1到第25帧的预测结果保留到一行数组
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])

        # i代表第i个动作类别
        errs[i] = ret_log
        all = all + time_takes
    ave_all = all / 30
    print("all=" + str(all))
    print("ave_all=" + str(ave_all))
    errs[-1] = np.mean(errs[:-1], axis=0)
    acts = np.expand_dims(np.array(acts + ["average"]), axis=1)
    # value = np.concatenate([acts, errs.astype(np.str)], axis=1)
    value = np.concatenate([acts, errs.astype(str)], axis=1)
    log.save_csv_log(opt, head, value, is_create=True, file_name='test_pre_action')


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
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
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    # kernel_size = 10，seq_in代表 M + T 中的 M ，
    seq_in = opt.kernel_size
    # 去掉不要的关节和相应的维度
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
    # 一次预测10帧，迭代3次预测25帧
    itera = 3
    # idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
    #         out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))

    # p3d_h36 维度为tensor(32, 75, 96), batch_size是32，取的一个 5(50+25) 是一个 ground_truth 样本大小
    for i, (p3d_h36) in enumerate(data_loader):
        # p3d_h36 [32, 75, 96]

        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue

        n += batch_size

        p3d_h36 = p3d_h36.float().cuda()
        p3d_src = p3d_h36.clone()[:, :, dim_used]
        # p3d_src [32, 75, 66]

        # input_n = 50, output_n = 10 , itera = 3
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=10, itera=itera)
        # 一次性预测75帧 (in 50 , out 25)

        p3d_out_all = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]
        # p3d_out_all [32, 25, 66] 取后25帧

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
        # p3d_out [32, 25, 96] 后25帧所有关节

        p3d_out = p3d_out.reshape([-1, out_n, 32, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])

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
