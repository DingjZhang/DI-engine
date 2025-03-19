import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class WeightAdjustmentAgent(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, lr=0.001):
        super(WeightAdjustmentAgent, self).__init__()
        
        # 策略网络(Actor)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 输出三个权重：w1, w2, w3
            nn.Softmax(dim=-1)  # 确保权重总和为1
        )
        
        # 价值网络(Critic)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 存储旧策略用于PPO更新
        self.old_actor_state_dict = None
        
    def get_weights(self, state):
        """根据当前状态生成目标权重(w1, w2, w3)"""
        return self.actor(state)
    
    def evaluate_states(self, states, actions=None):
        """评估状态价值和动作的对数概率"""
        values = self.critic(states)
        
        if actions is not None:
            weight_distributions = torch.distributions.Dirichlet(
                F.softplus(self.actor(states)) + 1e-5
            )
            log_probs = weight_distributions.log_prob(actions)
            entropy = weight_distributions.entropy().mean()
            return values, log_probs, entropy
        
        return values
    
    def backup_old_policy(self):
        """创建当前策略的副本"""
        self.old_actor_state_dict = self.actor.state_dict().copy()
    
    def restore_old_policy(self):
        """恢复旧策略"""
        self.actor.load_state_dict(self.old_actor_state_dict)
    
    def ppo_update(self, states, actions, advantages, returns, old_log_probs, epsilon=0.2, entropy_coef=0.01):
        """使用PPO算法更新策略和价值网络"""
        # 评估当前策略
        values, log_probs, entropy = self.evaluate_states(states, actions)
        
        # 计算策略比率
        ratios = torch.exp(log_probs - old_log_probs)
        
        # 计算surrogate损失
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
        
        # 计算actor和critic损失
        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
        critic_loss = F.mse_loss(values, returns)
        
        # 更新actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

def build_waa_state(system_metrics):
    """
    构建权重调整智能体的状态表示
    
    参数:
        system_metrics: 包含系统指标的字典
    
    返回:
        torch.Tensor: 状态张量
    """
    # 提取相关指标
    load_intensity = system_metrics['load_intensity']  # 负载强度(0-1)
    private_cloud_utilization = system_metrics['private_cloud_utilization']  # 私有云资源利用率(0-1)
    public_cloud_cost = system_metrics['public_cloud_cost']  # 标准化的公有云成本
    avg_latency = system_metrics['avg_latency']  # 标准化的平均延迟
    avg_jitter = system_metrics['avg_jitter']  # 标准化的抖动
    sla_violation_rate = system_metrics.get('sla_violation_rate', 0)  # SLA违反率(0-1)
    
    # 组合特征为状态向量
    state = torch.tensor([
        load_intensity,
        private_cloud_utilization,
        public_cloud_cost,
        avg_latency,
        avg_jitter,
        sla_violation_rate,
    ], dtype=torch.float32)
    
    return state

def calculate_waa_reward(weights, load_intensity, load_threshold=0.7):
    """
    计算权重调整智能体的奖励
    
    参数:
        weights: (w1, w2, w3)元组，表示延迟、抖动和成本的权重
        load_intensity: 标准化的负载强度(0到1)
        load_threshold: 区分低负载和高负载的阈值
    
    返回:
        float: 奖励值
    """
    w1, w2, w3 = weights  # 延迟、抖动、成本权重
    
    if load_intensity < load_threshold:
        # 低负载：优先考虑成本(w3应该更高)
        cost_priority = w3 / (w1 + w2 + 1e-6)
        return cost_priority
    else:
        # 高负载：优先考虑延迟和抖动(w1和w2应该更高)
        performance_priority = (w1 + w2) / (w3 + 1e-6)
        return performance_priority