
from utils import CMU_motion_3d as CMU_Motion3D
from model import AttModel_CMU
from utils.opt import Options
from utils import util
from utils import log
from torch.nn.parameter import Parameter

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print('>>> create models')
    in_features = opt.in_features  # 75
    d_model = opt.d_model  # 256
    kernel_size = opt.kernel_size  # 10
    # dct_n = 20
    # num_stage = 12
    net_pred = AttModel_CMU.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.is_load or opt.is_eval:
        model_path_len = '{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        # dataset = datasets.Datasets(opt, split=0)
        # actions = ["walking", "eating", "smoking", "discussion", "directions",
        #            "greeting", "phoning", "posing", "purchases", "sitting",
        #            "sittingdown", "takingphoto", "waiting", "walkingdog",
        #            "walkingtogether"]
        dataset = CMU_Motion3D.CMU_Motion3D(opt, split=0)
        # split=0:Train  split!=0:Test

        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        valid_dataset = CMU_Motion3D.CMU_Motion3D(opt, split=2)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

    test_loader = {}
    acts = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
            "washwindow"]
    for act in acts:
        test_dataset = CMU_Motion3D.CMU_Motion3D(opt=opt, split=2, actions=act)
        dim_used = dataset.dim_used
        test_loader[act] = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True)

    dim_used = dataset.dim_used

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, dim_used=dim_used)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        # print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt,
                                  dim_used=dim_used)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))

            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo, dim_used=dim_used)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))

            test_error = 0
            for act in acts:
                ret_test = run_model(net_pred, is_train=3, data_loader=test_loader[act], opt=opt, epo=epo,
                                     dim_used=dim_used)
                for j in range(1, 11):
                    test_error += ret_test["#{:d}ms".format(j * 40)]

            test_error = test_error / (10 * len(acts))
            print('testing error: {:.3f}'.format(test_error))

            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            # 保留测试集的测试结果
            if test_error < err_best:
                err_best = test_error
                is_best = True

            # 保留训练的模型
            log.save_ckpt({'epoch': epo,
                           'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)

def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, dim_used=None):
    # is_train :0训练 , 1验证, 3测试
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    # loss

    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = (np.array(range(opt.output_n)) + 1) * 40
        m_p3d_h36 = np.zeros([opt.output_n])

    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    # in_n:50
    # out_n:10
    # seq_in:10
    seq_in = opt.kernel_size
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 3
    # idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
    #         out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = Parameter(torch.FloatTensor(1)).to(device)
    b = Parameter(torch.FloatTensor(1)).to(device)

    for i, (p3d_h36) in enumerate(data_loader):
        # p3d_h36 Tensor (32, 60, 114)
        batch_size, seq_n, all_dim = p3d_h36.shape
        # batch_size 32  seq_n 60 all_dim:114
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()

        # dim_used = 75
        p3d_h36 = p3d_h36.float().cuda()
        # p3d_h36:[32, 60, 114]

        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        # p3d_sup (ground_truth) :Tensor(32, 20, 25, 3)

        p3d_src = p3d_h36.clone()[:, :, dim_used]
        # p3d_src :Tensor(32, 60, 75) Input of net

        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera)
        # p3d_out_all Tensor(32, 20, 1, 75)

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0]
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
        p3d_out = p3d_out.reshape([-1, out_n, all_dim // 3, 3])
        # p3d_out: Tensor(32, 10, 38, 3) prediction

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, all_dim // 3, 3])
        # Tensor(32, 60, 38, 3) ground_truth

        p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])
        # p3d_out_all :Tensor(32, 20, 1, 25, 3)

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            # Loss: 计算kernel_size + output_n长度的 ground_truth 与 输出之间的l2距离
            # p3d_out_all[:, :, 0] - p3d_sup Tensor(32, 20, 25, 3)
            p3d_out_all1 = p3d_out_all[:, :, 0]
            loss_p3d = torch.mean(torch.norm(p3d_out_all1 - p3d_sup, dim=3))

            p3d_all_smooth = p3d_out_all1[:, 1:, :, :] - p3d_out_all1[:, :-1, :, :]
            p3d_sup_smooth = p3d_sup[:, 1:, :, :] - p3d_sup[:, :-1, :, :]
            loss_vel = torch.mean(torch.norm(p3d_all_smooth - p3d_sup_smooth, dim=3))

            loss_all = a * loss_p3d + b * loss_vel
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:
            # if is validation or train simply output the overall mean error
            # 训练集和验证集
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            # 测试
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))
            # time()-bt: 当前这一轮下，处理一个batch的时间， time()-st 当前这轮训练的总时长
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            # j 表示 第几帧 m_p3d_h36 字典类型数据， 记录每一帧为索引的 MPJPE 值
            # titles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            ret["#{:d}ms".format(titles[j])] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()

    if option.is_eval == False:
        main(option)
    else:
        eval(option)
