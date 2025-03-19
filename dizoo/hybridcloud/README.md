# Hybrid Cloud Environment

## 概述

混合云环境（Hybrid Cloud Environment）是一个用于仿真混合云环境下微服务调度的多智能体强化学习环境。该环境模拟了一个由本地服务器集群和远程服务器集群组成的混合云架构，其中部署了多个相互依赖的微服务。

环境中有两个智能体：
1. 权重调整智能体（Weight Adjustment Agent, WAA）：负责根据系统状态动态调整目标权重（w1、w2、w3）
2. 微服务调度智能体（Microservice Scheduling Agent, MSA）：负责决定每个微服务容器的部署位置

## 环境特点

- **本地服务器集群**：CPU资源有限，但没有额外的访问延迟
- **远程服务器集群**：CPU资源无限，但存在访问延迟和使用费用
- **微服务应用**：包含多个微服务，它们之间有依赖关系，形成调用图
- **动态负载**：支持从文件读入负载数据，模拟真实的应用负载变化
- **性能指标**：包括延迟、抖动和成本三个主要指标

## 使用方法

### 安装依赖

```bash
pip install gymnasium numpy
```

### 基本使用

```python
from dizoo.hybridcloud.envs.hybrid_cloud import HybridCloudEnv

# 创建环境
env_config = {
    'num_local_servers': 5,
    'num_remote_servers': 5,
    'num_microservices': 10,
    'max_steps': 200,
    'remote_access_delay': 10.0,
    'remote_cost_per_cpu': 0.1,
    'load_file': None,  # 可以指定负载文件路径
}

env = HybridCloudEnv(env_config)

# 重置环境
obs = env.reset()

# 执行一步
action = {
    0: np.array([0.33, 0.33, 0.34]),  # WAA动作：权重调整
    1: np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # MSA动作：微服务部署位置
}
obs, reward, done, info = env.step(action)
```

### 使用MAPPO算法训练

```bash
# 使用命令行工具
ding -m serial_onpolicy -c dizoo/hybridcloud/config/hybrid_cloud_mappo_config.py -s 0

# 或者直接运行Python脚本
python dizoo/hybridcloud/entry/hybrid_cloud_mappo_main.py
```

## 环境参数

- `num_local_servers`：本地服务器数量
- `num_remote_servers`：远程服务器数量
- `num_microservices`：微服务数量
- `max_steps`：最大步数
- `remote_access_delay`：远程服务器访问延迟（毫秒）
- `remote_cost_per_cpu`：远程服务器CPU使用费用
- `load_file`：负载文件路径，每行一个数字表示每秒请求数

## 观察空间

- **WAA观察**：[负载，本地利用率，远程利用率，延迟，抖动，成本]
- **MSA观察**：[微服务特征，调用图，当前部署，权重]

## 动作空间

- **WAA动作**：连续值 [w1, w2, w3]，表示延迟、抖动和成本的权重
- **MSA动作**：离散值，表示每个微服务的部署位置（本地或远程服务器）

## 奖励函数

- **WAA奖励**：基于权重分配与系统负载的匹配程度
- **MSA奖励**：负的加权目标值 -(w1×延迟 + w2×抖动 + w3×成本)