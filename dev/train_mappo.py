import random
from collections import deque
import torch.nn.utils as nn_utils

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
    """
    计算广义优势估计(GAE)
    
    参数:
        rewards: 奖励列表
        values: 价值估计列表
        next_values: 下一状态价值估计列表
        dones: 完成标志列表
        gamma: 折扣因子
        lambda_: GAE参数
    
    返回:
        advantages: 计算的优势
        returns: 计算的回报
    """
    advantages = []
    gae = 0
    
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = next_values[i]
        else:
            next_value = values[i + 1]
        
        # 计算TD误差
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        
        # 计算GAE
        gae = delta + gamma * lambda_ * (1 - dones[i]) * gae
        
        # 在开头插入优势
        advantages.insert(0, gae)
    
    # 计算回报
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    return torch.tensor(advantages), torch.tensor(returns)

def train_mappo(env, waa, msa, num_episodes=1000, steps_per_episode=100, 
               ppo_epochs=10, batch_size=64, gamma=0.99, lambda_=0.95, 
               max_grad_norm=0.5, epsilon=0.2, entropy_coef=0.01):
    """
    使用Multi-Agent PPO训练两个智能体
    
    参数:
        env: 环境
        waa: 权重调整智能体
        msa: 微服务调度智能体
        num_episodes: 训练的回合数
        steps_per_episode: 每回合的最大步数
        ppo_epochs: PPO更新的迭代次数
        batch_size: 批量大小
        gamma: 折扣因子
        lambda_: GAE参数
        max_grad_norm: 梯度裁剪的最大范数
        epsilon: PPO裁剪参数
        entropy_coef: 熵系数
    
    返回:
        list: 训练统计数据
    """
    stats = {'waa_rewards': [], 'msa_rewards': [], 'weighted_objectives': []}
    
    for episode in range(num_episodes):
        # 重置环境
        obs = env.reset()
        
        # 获取初始状态表示
        waa_state = build_waa_state(obs['system_metrics'])
        g, node_features, edge_features = build_microservice_graph(
            obs['microservices'], obs['dependencies'])
        
        # 存储回合数据的列表
        waa_states = []
        waa_actions = []
        waa_rewards = []
        waa_log_probs = []
        waa_values = []
        waa_dones = []
        
        msa_graphs = []
        msa_node_features = []
        msa_edge_features = []
        msa_weights = []
        msa_actions = []
        msa_rewards = []
        msa_log_probs = []
        msa_values = []
        msa_dones = []
        
        episode_waa_reward = 0
        episode_msa_reward = 0
        episode_weighted_obj = 0
        
        for step in range(steps_per_episode):
            # WAA选择权重
            with torch.no_grad():
                weights = waa.get_weights(waa_state)
                waa_value = waa.evaluate_states(waa_state)
                
                # 创建概率分布
                weight_dist = torch.distributions.Dirichlet(F.softplus(waa.actor(waa_state)) + 1e-5)
                waa_log_prob = weight_dist.log_prob(weights)
            
            # MSA选择部署位置
            with torch.no_grad():
                action_logits, msa_value = msa(g, node_features, edge_features, weights, training=True)
                
                # 创建概率分布
                action_dist = torch.distributions.Categorical(logits=action_logits)
                actions = action_dist.sample()
                msa_log_prob = action_dist.log_prob(actions)
            
            # 在环境中执行动作
            next_obs, rewards, done, _ = env.step(weights.numpy(), actions.numpy())
            
            # 获取下一状态表示
            next_waa_state = build_waa_state(next_obs['system_metrics'])
            next_g, next_node_features, next_edge_features = build_microservice_graph(
                next_obs['microservices'], next_obs['dependencies'])
            
            # 获取奖励
            waa_reward = rewards['waa_reward']
            msa_reward = rewards['msa_reward']
            weighted_objective = -msa_reward  # MSA奖励的负值
            
            # 存储经验
            waa_states.append(waa_state)
            waa_actions.append(weights)
            waa_rewards.append(waa_reward)
            waa_log_probs.append(waa_log_prob)
            waa_values.append(waa_value.item())
            waa_dones.append(done)
            
            msa_graphs.append(g)
            msa_node_features.append(node_features)
            msa_edge_features.append(edge_features)
            msa_weights.append(weights)
            msa_actions.append(actions)
            msa_rewards.append(msa_reward)
            msa_log_probs.append(msa_log_prob)
            msa_values.append(msa_value.mean().item())
            msa_dones.append(done)
            
            # 更新状态
            waa_state = next_waa_state
            g, node_features, edge_features = next_g, next_node_features, next_edge_features
            
            # 累积奖励
            episode_waa_reward += waa_reward
            episode_msa_reward += msa_reward
            episode_weighted_obj += weighted_objective
            
            if done:
                break
        
        # 获取最终价值估计以进行GAE计算
        with torch.no_grad():
            final_waa_value = waa.evaluate_states(waa_state).item()
            final_msa_value = msa(g, node_features, edge_features, weights, training=True)[1].mean().item()
        
        # 计算WAA优势和回报
        waa_next_values = waa_values[1:] + [final_waa_value]
        waa_advantages, waa_returns = compute_gae(
            waa_rewards, waa_values, waa_next_values, waa_dones, gamma, lambda_)
        
        # 计算MSA优势和回报
        msa_next_values = msa_values[1:] + [final_msa_value]
        msa_advantages, msa_returns = compute_gae(
            msa_rewards, msa_values, msa_next_values, msa_dones, gamma, lambda_)
        
        # 将列表转换为张量
        waa_states = torch.stack(waa_states)
        waa_actions = torch.stack(waa_actions)
        waa_log_probs = torch.stack(waa_log_probs)
        
        # 存储回合统计数据
        stats['waa_rewards'].append(episode_waa_reward)
        stats['msa_rewards'].append(episode_msa_reward)
        stats['weighted_objectives'].append(episode_weighted_obj)
        
        print(f"回合 {episode+1}/{num_episodes}, WAA奖励: {episode_waa_reward:.4f}, "
              f"MSA奖励: {episode_msa_reward:.4f}, 加权目标: {episode_weighted_obj:.4f}")
        
        # 执行PPO更新
        for epoch in range(ppo_epochs):
            # 备份旧策略
            waa.backup_old_policy()
            msa.backup_old_policy()
            
            # 采样随机小批量
            indices = torch.randperm(len(waa_states))
            for start_idx in range(0, len(waa_states), batch_size):
                # 获取小批量索引
                idx = indices[start_idx:start_idx + batch_size]
                
                # WAA更新
                waa_batch_states = waa_states[idx]
                waa_batch_actions = waa_actions[idx]
                waa_batch_advantages = waa_advantages[idx]
                waa_batch_returns = waa_returns[idx]
                waa_batch_log_probs = waa_log_probs[idx]
                
                # 执行WAA更新
                waa_actor_loss, waa_critic_loss = waa.ppo_update(
                    waa_batch_states, waa_batch_actions, waa_batch_advantages, 
                    waa_batch_returns, waa_batch_log_probs, epsilon, entropy_coef)
                
                # MSA更新 - 单独更新每个图
                for i in idx:
                    msa_batch_graph = msa_graphs[i]
                    msa_batch_node_features = msa_node_features[i]
                    msa_batch_edge_features = msa_edge_features[i]
                    msa_batch_weights = msa_weights[i]
                    msa_batch_actions = msa_actions[i]
                    msa_batch_advantages = msa_advantages[i]
                    msa_batch_returns = msa_returns[i]
                    msa_batch_log_probs = msa_log_probs[i]
                    
                    # 执行MSA更新
                    msa_actor_loss, msa_critic_loss = msa.ppo_update(
                        msa_batch_graph, msa_batch_node_features, msa_batch_edge_features,
                        msa_batch_weights, msa_batch_actions, msa_batch_advantages, 
                        msa_batch_returns, msa_batch_log_probs, epsilon, entropy_coef)
        
        # 如果需要，可以在这里保存模型检查点
        if episode % 100 == 0:
            torch.save(waa.state_dict(), f"models/waa_model_ep{episode}.pth")
            torch.save(msa.state_dict(), f"models/msa_model_ep{episode}.pth")
    
    # 保存最终模型
    torch.save(waa.state_dict(), "models/waa_model_final.pth")
    torch.save(msa.state_dict(), "models/msa_model_final.pth")
    
    return stats