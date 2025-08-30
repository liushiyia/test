import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pickle

def plot_action_statistics(data_dir, cur_dir, show=True):
    # 构建文件路径
    base_path = Path(data_dir) / cur_dir
    velocities_file = base_path / f"maddpg_statistics.pkl"

    # 加载速度统计量数据
    with open(velocities_file, 'rb') as f:
        vel_statistics = pickle.load(f)

    agents = ['agent_0', 'agent_1', 'agent_2']
    num_agents = len(agents)

    # 创建保存目录
    plot_dir = base_path
    plt.figure(figsize=(6 * num_agents, 12))
    for agent_idx, stats in vel_statistics.items():
        if agent_idx == 'agent_0':
            i = 0
        elif agent_idx == 'agent_1':
            i = 1
        else:
            i = 2
        steps = list(range(len(stats['mean_x'])))
        # 均值变化曲线图
        plt.subplot(3, num_agents, i + 1)
        plt.plot(steps, stats['mean_x'], 'r-', label='x mean')
        plt.plot(steps, stats['mean_y'], 'b-', label='y  mean')
        plt.title(f'agents{i}  Means')
        plt.ylabel('Means Values')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 标准差变化曲线
        plt.subplot(3, num_agents, num_agents + i + 1)
        plt.plot(steps, stats['std_x'], 'r-', label='x std')
        plt.plot(steps, stats['std_y'], 'b-', label='y std')
        plt.title(f'agents{i} std')
        plt.ylabel('Std Values')
        plt.legend()

        # 对数标准差变化曲线
        plt.subplot(3, num_agents, num_agents * 2 + i + 1)
        plt.plot(steps, stats['logstd_x'], 'r-', label='x logstd')
        plt.plot(steps, stats['logstd_y'], 'b-', label='y logstd')
        plt.title(f'agents{i} logstd')
        plt.ylabel('LogStd Values')
        plt.legend()

    # 图像保存
    save_path = plot_dir / f"agent_vel_statistics.png"
    plt.savefig(save_path)
    if show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot maddpg agent Losses')
    parser.add_argument("--data_dir", type=str, default= "/home/zyw/code/Critic_Actor_tf2/results/maddpg/learning_curves", help='Directory containing data')
    parser.add_argument("--cur_dir", type=str, default="2025-08-14-03-59-56")
    parser.add_argument("--show", action='store_true')

    args = parser.parse_args()
    plot_action_statistics(args.data_dir, args.cur_dir, args.show)
