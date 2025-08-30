import numpy as np
from pyglet.window import key

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input通过键盘控制智能体移动的策略类
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    """
    方向键控制移动：LEFT/RIGHT/UP/DOWN。

    动作模式分为两种：

    离散动作：每个方向对应一个固定编号（如LEFT=1）。

    连续动作：用5维向量表示，各维度对应不同方向的速度。

    适用于人工控制智能体的强化学习演示环境。

    可用于调试环境或录制人类操作数据。
    """
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 1.0
            if self.move[1]: u[2] += 1.0
            if self.move[3]: u[3] += 1.0
            if self.move[2]: u[4] += 1.0
            if True not in self.move:
                u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True
    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False
