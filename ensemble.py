import torch
'''
find命令查找指定pt文件，并重定向到txt文件中，保留最后10个pt文件：
find checkpoints/vsr_cmlr -name iter_??.pt | tail -n 10 > last_10_pt.txt
'''

def ensemble(ckpt_path, model_path='model_avg_10.pt'):
    '''
    :param ckpt_path: checkpoint保存路径的文件，每行一个checkpoint路径
    :param model_path: 最终融合的model保存的路径
    :return:
    '''
    avg = None
    N = 0
    with open(ckpt_path, 'r') as f:
        for path in f:
            print(path.strip())
            N += 1
            # load model state_dict
            states = torch.load(path.strip(), map_location=lambda storage, loc: storage, weights_only=True)['model']
            if avg is None:
                avg = states
            else:
                for k in avg.keys():
                    avg[k] += states[k]
    print('num of ckpt:', N)
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= N
            else:
                avg[k] //= N
    torch.save(avg, model_path)
    print(f'{model_path} saved !!')


#ensemble('last_10_grid.txt', 'grid_avg_rvq.pt2')
#ensemble('last_10_grid.txt', 'grid_avg_10.pt5')
#ensemble('last_10_grid.txt', 'grid_avg10_baseline.pt2')
#ensemble('last_10_grid.txt', 'grid_avg_nolipmask.pt')
#ensemble('last_10_grid2.txt', 'grid_avg_10.pt4')
#ensemble('last_10_grid3.txt', 'grid_avg_10.pt3')
ensemble('last_10_grid.txt', 'grid_avg_10.pt')

#ensemble('last_10_cmlr.txt', 'cmlr_avg_10.pt')
#ensemble('last_10_cmlr.txt', 'cmlr_avg_10.pt6')
#ensemble('last_10_cmlr.txt', 'cmlr_avg_10_2stage.pt4')
#ensemble('last_10_cmlr2.txt', 'cmlr_avg_10_2stage.pt3')
#ensemble('last_10_cmlr.txt', 'cmlr_avg10_baseline.pt')
#ensemble('last_10_baseline.txt', 'model_avg_baseline2.pt')
#ensemble('last_10_baseline.txt', 'model_avg_baseline3.pt')
