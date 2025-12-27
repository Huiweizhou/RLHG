"""
实现了数据预处理，主要用于生成状态-动作空间并将其保存到文件中。
"""
import pickle
import os
import argparse
from src.environment import Env
from baseDataset import baseDataset
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocess', usage='preprocess_data.py [<args>] [-h | --help]')
    parser.add_argument('--data_dir', default='data/Virology', type=str, help='Path to data.')
    parser.add_argument('--outfile', default='state_actions_space.pkl', type=str, help='file to save the preprocessed data.')
    parser.add_argument('--store_actions_num', default=0, type=int, help='maximum number of stored neighbors, 0 means store all.')
    args = parser.parse_args()

    # 构建文件路径
    trainF = os.path.join(args.data_dir, 'train.txt')
    testF = os.path.join(args.data_dir, 'test.txt')
    statF = os.path.join(args.data_dir, 'stat.txt')
    validF = os.path.join(args.data_dir, 'valid.txt')
    if not os.path.exists(validF):
        validF = None

    # 初始化数据集
    dataset = baseDataset(trainF, testF, statF, validF)
    config = {
        'num_rel': dataset.num_r,
        'num_ent': dataset.num_e,
    }
    # 创建环境实例
    env = Env(dataset.allQuadruples, config)
    state_actions_space = {}  # 初始化状态-动作空间字典
    timestamps = list(dataset.get_all_timestamps())  # 获取所有时间戳
    print(args)

    # 使用 tqdm 创建进度条
    with tqdm(total=len(dataset.allQuadruples)) as bar:
        for (head, rel, tail, t) in dataset.allQuadruples:
            if (head, t, True) not in state_actions_space.keys():
                state_actions_space[(head, t, True)], _ = env.get_state_actions_space_complete(head, t, True, args.store_actions_num)
                state_actions_space[(head, t, False)], _ = env.get_state_actions_space_complete(head, t, False, args.store_actions_num)
            if (tail, t, True) not in state_actions_space.keys():
                state_actions_space[(tail, t, True)] = env.get_state_actions_space_complete(tail, t, True, args.store_actions_num)
                state_actions_space[(tail, t, False)] = env.get_state_actions_space_complete(tail, t, False, args.store_actions_num)
            bar.update(1)
    # 将状态-动作空间保存为 pickle 文件
    pickle.dump(state_actions_space, open(os.path.join(args.data_dir, args.outfile), 'wb'))
