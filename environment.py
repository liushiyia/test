"""
多智能体环境实现，基于OpenAI Gym框架。包含核心环境类MultiAgentEnv和批量环境包装类BatchMultiAgentEnv。
适用于多智能体强化学习场景，支持离散/连续动作空间、通信机制、协作/竞争任务。
"""
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete  # 自定义多维离散空间实现
from multiagent.scenarios.formation import Scenario

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
# 该环境不考虑新加入/有损坏智能体情况（智能体环境固定）
class MultiAgentEnv(gym.Env):
    """
    多智能体环境基类，管理多个智能体的交互逻辑

    属性：
    world: 物理世界对象，包含所有实体和物理规则
    agents: 当前策略控制的智能体列表
    n: 智能体数量
    action_space: 每个智能体的动作空间列表
    observation_space: 每个智能体的观察空间列表
    """
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None,  done_callback=None,
                 info_callback=None, shared_viewer=True):

        """
        初始化多智能体环境

        参数：
        world: 物理世界对象
        reset_callback: 重置环境回调函数
        reward_callback: 计算奖励回调函数
        observation_callback: 生成观察值回调函数
        info_callback: 生成调试信息回调函数
        done_callback: 判断episode结束回调函数
        shared_viewer: 是否共享渲染视图
        """

        # 1.环境核心组件
        self.world = world
        self.agents = self.world.policy_agents  # 获取需要学习的策略智能体
        # set required vectorized gym env property
        self.n = len(world.policy_agents)  # 智能体数量

        # 2.scenario callbacks场景回调函数配置
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # 3.environment parameters环境参数配置
        self.discrete_action_space = False # 是否使用离散动作空间

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False  # 输入是否为离散形式

        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False  # 强制离散化连续动作

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False   # 是否共享奖励（协作模式）
        self.time = 0

        # 4.configure spaces初始化动作和观测空间
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            # -----------------------4.1 构建动作空间 ----------------------#
            total_action_space = [] # 初始化动作空间容器
            #------------------------4.1.1 物理动作空间---------------------#
            # physical action space物理动作空间（移动)
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)  # 离散动作：无操作+各方向移动
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)  # 连续动作空间
            if agent.movable:  # 如果智能体可移动
                total_action_space.append(u_action_space)

            # --------------------4.1.2 communication action space通信动作空间-----#
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)  # 离散通信动作
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)  # 连续通信动作
            if not agent.silent:  # 如果智能体可通信
                total_action_space.append(c_action_space)

            # -------------------4.1.3 total action space合并动作空间------------#
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]): # 离散型动作
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else: # 连续型动作
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # -------------4.2 observation space观测空间-----------------------#
            obs_dim = len(observation_callback(agent, self.world))
            # print(f"Observation for agent {agent.name}: {observation_callback(agent, self.world)}")
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering渲染初始化
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        """
        执行环境步进

        参数：
        action_n: 所有智能体的动作列表

        返回：
        obs_n: 新的观察值列表
        reward_n: 奖励值列表
        done_n: episode结束标记列表
        info_n: 调试信息字典
        """

        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent设置每个智能体的动作
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state更新状态
        self.world.step()
        # # TODO FOR add USV model start4:
        # self.world.integrate_state()
        # # TODO FOR add USV model end
        # record observation for each agent收集所有智能体的信息
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case处理协作模式的共享奖励
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        """重置环境状态"""

        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        """获取指定智能体的调试信息"""
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        """获取指定智能体的观测值"""
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        """判断是否需要结束当前episode"""
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        """计算单个智能体的奖励"""
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        """
        解析并设置智能体动作

        处理不同动作空间类型：
        - MultiDiscrete: 多维离散动作
        - Tuple: 混合动作空间
        - Box: 连续动作空间
        """

        agent.action.u = np.zeros(self.world.dim_p)  # 初始化物理动作
        agent.action.c = np.zeros(self.world.dim_c)  # 初始化通信动作
        # process action等价于normalization操作
        if isinstance(action_space, MultiDiscrete):  # 处理多维离散动作
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else: # 连续型动作
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:  # 如果动作输入是离散的
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action  # 处理离散动作（通过索引值判断方向）
                if action[0] == 1: agent.action.u[0] = -1.0 #左
                if action[0] == 2: agent.action.u[0] = +1.0 #右
                if action[0] == 3: agent.action.u[1] = -1.0 #下
                if action[0] == 4: agent.action.u[1] = +1.0 #上
            else:
                if self.force_discrete_action:# 如果需要将连续动作强制转换为离散
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space: # 如果物理动作空间本身是离散的
                    # 基于 one-hot 编码设置 x 和 y 方向的动作值
                    agent.action.u[0] += action[0][1] - action[0][2]  # 左（1） - 右（2）
                    agent.action.u[1] += action[0][3] - action[0][4]  # 下（3） - 上（4）
                else: # 连续性动作
                    # # TODO FOR add USV model start3:
                    agent.action.u = action[0]   # 纵向速度指令u_c
                    # agent.action.u[1] = action[1]   # 艏摇角速度指令r_c
                    # # TODO FOR add USV model end
            sensitivity = 1.0   # 默认加速度因子 原始值：5.0
            if agent.accel is not None:   # 如果智能体设置了加速度值
                sensitivity = agent.accel
            agent.action.u *= sensitivity # 缩放物理动作
            action = action[1:] # 移除已经使用的动作部分（物理动作）
        if not agent.silent:  # 如果智能体不是静默的（可以通信）
            # communication action
            if self.discrete_action_input:  # 如果通信动作是离散输入
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0   # 使用 one-hot 编码表示通信信息
            else:
                agent.action.c = action[0]  # 否则直接使用连续通信向量
            action = action[1:]  # 移除已使用的通信动作部分
        # make sure we used all elements of action确保 action 中的所有部分都已被处理
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments向量化封装器，用于处理一批多智能体环境
# assumes all environments have the same observation and action space假设所有环境具有相同的 observation 和 action space
class BatchMultiAgentEnv(gym.Env):
    """
    批量多智能体环境包装器，用于并行运行多个环境实例

    功能：
    - 将多个独立环境实例组合成批量环境
    - 实现同步执行多个环境的step/reset/render操作
    - 假设所有子环境具有相同的动作空间和观察空间

    属性：
    env_batch: 包含多个MultiAgentEnv实例的列表
    """
    metadata = {
        'runtime.vectorized': True, # 声明支持矢量化环境
        'render.modes' : ['human', 'rgb_array'] # 支持的渲染模式
    }

    def __init__(self, env_batch):
        """
        初始化批量环境

        参数：
        env_batch (list): 包含多个MultiAgentEnv实例的列表，要求所有环境具有：
                         - 相同的动作空间结构
                         - 相同的观察空间结构
                         - 相同的智能体数量（可选，根据具体使用场景）
        """
        self.env_batch = env_batch

    @property
    def n(self):
        """获取所有环境中智能体的总数（批量总智能体数）"""
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        """获取动作空间（假设所有子环境动作空间相同）"""
        return self.env_batch[0].action_space # 取第一个环境的动作空间作为代表

    @property
    def observation_space(self):
        """获取观察空间（假设所有子环境观察空间相同）"""
        return self.env_batch[0].observation_space # 取第一个环境的观察空间作为代表

    def step(self, action_n, time):
        """
        同步执行所有子环境的一个时间步

        参数：
        action_n (list): 包含所有智能体动作的扁平化列表，结构为：
                        [env1_agent1_action, env1_agent2_action..., env2_agent1_action...]
        time: 当前时间步（具体用法取决于子环境实现）

        返回：
        obs_n: 所有智能体的新观察值列表（扁平化）
        reward_n: 所有智能体的奖励值列表（扁平化）
        done_n: 所有环境的结束标志列表（每个环境一个标志）
        info_n: 调试信息字典（当前简单实现，可根据需求扩展）
        """
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0 # 动作列表索引指针
        for env in self.env_batch:
            # 执行子环境step（注意：原代码time参数未传递，可能需要调整）
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        """
       重置所有子环境

       返回：
       obs_n: 所有智能体的初始观察值列表（扁平化）
       """
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        """
        渲染所有子环境

        参数：
        mode: 渲染模式，可选'human'或'rgb_array'
        close: 是否关闭渲染窗口（实际各子环境可能忽略此参数）

        返回：
        results_n: 包含所有子环境渲染结果的列表
                  'human'模式可能返回None列表
                  'rgb_array'模式返回各环境图像数组
        """
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
