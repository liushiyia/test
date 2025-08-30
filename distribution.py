# -*- coding: utf-8 -*-
# @Date       : 2024/5/2417:31
# @Author     : Wang.zr
# @File name  : distribution.py
# @Description:
import tensorflow as tf


@tf.function
def gen_action_for_discrete(actions):
    """
    离散动作空间采样函数（使用Gumbel-Softmax技巧）

    功能：通过可导的方式从离散动作概率分布中采样，适用于分类动作选择
    应用场景：强化学习中的离散动作策略（如DQN、Policy Gradient）

    参数：
    actions: 神经网络的原始输出logits（未归一化的概率）

    数学原理：
    1. 生成Gumbel噪声：tf.math.log(-tf.math.log(u))， u~Uniform(0,1)
    2. 计算带噪声的logits：logits + Gumbel噪声：actions - tf.math.log(-tf.math.log(u))
    3. 通过softmax获得可导的近似one-hot向量

    输出：形状与输入相同的概率分布（可视为"软化"的one-hot向量）
    很本目的：将Actor网络输出进行归一化
    """
    # 创建一个与 actions 同形状的张量 u，其中的元素是从 均匀分布 U(0, 1) 中采样得到的。
    u = tf.random.uniform(tf.shape(actions), dtype=tf.float64) #u 是用于构造 Gumbel 噪声 的基础
    return tf.nn.softmax(actions - tf.math.log(-tf.math.log(u)), axis=-1) # 获得可导的近似离散样本


@tf.function
def gen_action_for_continuous(actions):
    mean, logstd = tf.split(axis=1, num_or_size_splits=2, value=actions)
    std = tf.exp(logstd)
    noise = tf.random.normal(tf.shape(mean), dtype=tf.float64)

    v = mean + std * noise
    return tf.clip_by_value(v, -10.0, 10.0), mean, logstd, std    # 返回动作值和统计量
    # return v, mean, logstd, std  # 返回动作值和统计量
    # # TODO FOR add USV model start5:
    # """为USV生成连续动作：纵向速度u和艏摇角速度r"""
    # # 将输出分为u和r的两组(均值, 对数标准差)
    # u_mean, u_logstd, r_mean, r_logstd = tf.split(actions, 4, axis=1)
    #
    # # 计算标准差
    # u_std = tf.exp(u_logstd)
    # r_std = tf.exp(r_logstd)
    #
    # # 生成噪声
    # u_noise = tf.random.normal(tf.shape(u_mean), dtype=tf.float64)
    # r_noise = tf.random.normal(tf.shape(r_mean), dtype=tf.float64)
    #
    # # 生成动作
    # u = u_mean + u_std * u_noise
    # r = r_mean + r_std * r_noise
    #
    # # 裁剪到合理范围
    # u = tf.clip_by_value(u, -10.0, 10.0)
    # r = tf.clip_by_value(r, -2.0, 2.0)
    #
    # return tf.concat([u, r], axis=1), u_mean, u_std, r_mean, r_std
    # # TODO FOR add USV model end