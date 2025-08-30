import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pickle



def plot_agent_losses(data_dir, exp_name, cur_dir, show=True):
    # 构建文件路径
    base_path = Path(data_dir) / cur_dir
    critic_file = base_path / f"maddpg_critic_loss_{cur_dir}.pkl"
    actor_file = base_path / f"maddpg_actor_loss_{cur_dir}.pkl"

    # 加载损失数据
    with open(critic_file, 'rb') as f:
        critic_losses = pickle.load(f)
    with open(actor_file, 'rb') as f:
        actor_losses = pickle.load(f)

    agents = ['agent_0', 'agent_1', 'agent_2']
    num_agents = len(agents)

    # 创建专业图表
    plt.figure(figsize=(6 * num_agents, 8))

    # 为每个智能体创建两列子图
    for i in range(num_agents):
        # Critic损失子图
        plt.subplot(2, num_agents, i + 1)
        plt.plot(critic_losses[i], 'b-', label='Critic Loss', linewidth=2)
        plt.title(f'agents{i}')
        plt.xlabel('Training steps')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Actor损失子图
        plt.subplot(2, num_agents, num_agents + i + 1)
        plt.plot( actor_losses[i], 'r-', label='Actor Loss', linewidth=2)
        # plt.title(f'agents{i}-Actor Loss')
        plt.xlabel('Training steps')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

    # 创建保存目录
    plot_dir = base_path
    # plot_dir.mkdir(parents=True, exist_ok=True)

    # 图像保存
    save_path = plot_dir / f"losses.png"
    plt.savefig(save_path)
    if show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot maddpg agent Losses')
    parser.add_argument("--data_dir", type=str, default= "/home/zyw/code/Critic_Actor_tf2/results/maddpg/learning_curves", help='Directory containing data')
    parser.add_argument("--exp_name", type=str, default="maddpg")
    parser.add_argument("--cur_dir", type=str, default="2025-08-18-19-43-47")
    parser.add_argument("--show", action='store_true')

    args = parser.parse_args()
    plot_agent_losses(args.data_dir, args.exp_name, args.cur_dir, args.show)
