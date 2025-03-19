import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.nn import GATConv

class MicroserviceSchedulingAgent(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64, num_heads=4, 
                 num_placement_options=2, lr=0.001):
        super(MicroserviceSchedulingAgent, self).__init__()
        
        # 图注意力层
        self.gat1 = GATConv(
            in_feats=node_feat_dim, 
            out_feats=hidden_dim,
            num_heads=num_heads,
            feat_drop=0.2,
            attn_drop=0.2,
            residual=True,
            activation=F.elu
        )
        
        self.gat2 = GATConv(
            in_feats=hidden_dim * num_heads, 
            out_feats=hidden_dim,
            num_heads=1,
            feat_drop=0.2,
            attn_drop=0.2,
            residual=True,
            activation=F.elu
        )
        
        # 边特征处理
        self.edge_embed = nn.Linear(edge_feat_dim, hidden_dim)
        
        # 全局上下文集成
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for weights w1, w2, w3
            nn.ReLU()
        )
        
        # 策略头(actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_placement_options)
        )
        
        # 价值头(critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优化器
        self.actor_optimizer = optim.Adam([
            {'params': self.gat1.parameters()},
            {'params': self.gat2.parameters()},
            {'params': self.edge_embed.parameters()},
            {'params': self.global_context.parameters()},
            {'params': self.actor.parameters()}
        ], lr=lr)
        
        self.critic_optimizer = optim.Adam([
            {'params': self.gat1.parameters()},
            {'params': self.gat2.parameters()},
            {'params': self.edge_embed.parameters()},
            {'params': self.global_context.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)
        
        # 存储旧策略
        self.old_policy_state_dict = None
    
    def forward(self, g, node_features, edge_features, weights, training=False):
        """
        MSA的前向传播
        
        参数:
            g: 表示微服务调用关系的DGL图
            node_features: 节点特征张量
            edge_features: 边特征张量
            weights: 来自WAA的目标权重(w1, w2, w3)
            training: 是否处于训练模式
        
        返回:
            action_probs: 每个微服务的动作概率
            values: 价值估计
        """
        # 处理边特征
        edge_embeddings = self.edge_embed(edge_features)
        g.edata['ef'] = edge_embeddings
        
        # 第一个GAT层
        h = self.gat1(g, node_features)
        h = h.view(h.shape[0], -1)  # 合并多头注意力
        
        # 聚合边特征到节点
        g.ndata['h'] = h
        g.update_all(fn.copy_e('ef', 'm'), fn.mean('m', 'agg_ef'))
        h = h + g.ndata['agg_ef']
        
        # 第二个GAT层
        h = self.gat2(g, h).squeeze(1)
        
        # 广播权重到所有节点
        batch_size = g.batch_size if hasattr(g, 'batch_size') else 1
        weight_broadcast = weights.repeat(g.number_of_nodes() // batch_size, 1)
        
        # 连接节点嵌入和权重
        h_combined = torch.cat([h, weight_broadcast], dim=1)
        
        # 全局上下文集成
        h_global = self.global_context(h_combined)
        
        # Actor: 生成动作概率
        action_logits = self.actor(h_global)
        
        # Critic: 估计价值
        values = self.critic(h_global)
        
        if training:
            return action_logits, values
        else:
            # 推理期间，返回动作概率
            action_probs = F.softmax(action_logits, dim=1)
            return action_probs, values
    
    def backup_old_policy(self):
        """创建当前策略的副本"""
        self.old_policy_state_dict = {}
        self.old_policy_state_dict['gat1'] = self.gat1.state_dict().copy()
        self.old_policy_state_dict['gat2'] = self.gat2.state_dict().copy()
        self.old_policy_state_dict['edge_embed'] = self.edge_embed.state_dict().copy()
        self.old_policy_state_dict['global_context'] = self.global_context.state_dict().copy()
        self.old_policy_state_dict['actor'] = self.actor.state_dict().copy()
    
    def restore_old_policy(self):
        """恢复旧策略"""
        self.gat1.load_state_dict(self.old_policy_state_dict['gat1'])
        self.gat2.load_state_dict(self.old_policy_state_dict['gat2'])
        self.edge_embed.load_state_dict(self.old_policy_state_dict['edge_embed'])
        self.global_context.load_state_dict(self.old_policy_state_dict['global_context'])
        self.actor.load_state_dict(self.old_policy_state_dict['actor'])
    
    def evaluate_actions(self, g, node_features, edge_features, weights, actions):
        """评估动作的对数概率和价值"""
        action_logits, values = self.forward(g, node_features, edge_features, weights, training=True)
        
        # 创建分类分布
        action_dist = torch.distributions.Categorical(logits=action_logits)
        
        # 获取对数概率
        log_probs = action_dist.log_prob(actions)
        
        # 计算熵
        entropy = action_dist.entropy().mean()
        
        return log_probs, values, entropy
    
    def ppo_update(self, g, node_features, edge_features, weights, actions, advantages, returns, old_log_probs, 
                 epsilon=0.2, entropy_coef=0.01):
        """使用PPO算法更新策略和价值网络"""
        # 评估当前策略
        log_probs, values, entropy = self.evaluate_actions(g, node_features, edge_features, weights, actions)
        
        # 计算策略比率
        ratios = torch.exp(log_probs - old_log_probs)
        
        # 计算surrogate损失
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
        
        # 计算actor和critic损失
        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
        critic_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # 更新actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        # 更新critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

def build_microservice_graph(microservices, dependencies):
    """
    构建表示微服务调用关系的DGL图
    
    参数:
        microservices: 包含微服务及其属性的列表
        dependencies: 微服务之间的依赖关系(边)列表
    
    返回:
        g: DGL图
        node_features: 节点特征张量
        edge_features: 边特征张量
    """
    # 创建图
    src_nodes = [dep['source'] for dep in dependencies]
    dst_nodes = [dep['target'] for dep in dependencies]
    g = dgl.graph((src_nodes, dst_nodes))
    
    # 添加自环以确保消息传递给自身
    g = dgl.add_self_loop(g)
    
    # 准备节点特征
    node_features = []
    for ms in microservices:
        # 提取相关特征
        cpu_req = ms['cpu_requirement']
        mem_req = ms['memory_requirement']
        net_req = ms['network_requirement']
        current_load = ms['current_load']
        current_latency = ms.get('current_latency', 0)
        current_jitter = ms.get('current_jitter', 0)
        
        # 组合特征
        node_feat = torch.tensor([
            cpu_req, mem_req, net_req, current_load, 
            current_latency, current_jitter
        ], dtype=torch.float32)
        
        node_features.append(node_feat)
    
    # 转换为张量
    node_features = torch.stack(node_features)
    
    # 准备边特征
    edge_features = []
    for dep in dependencies:
        # 提取相关特征
        call_frequency = dep['call_frequency']
        data_volume = dep['data_volume']
        current_latency = dep.get('current_latency', 0)
        
        # 组合特征
        edge_feat = torch.tensor([
            call_frequency, data_volume, current_latency
        ], dtype=torch.float32)
        
        edge_features.append(edge_feat)
    
    # 添加自环边特征
    for _ in range(g.number_of_nodes()):
        # 自环特征
        self_edge_feat = torch.tensor([0, 0, 0], dtype=torch.float32)
        edge_features.append(self_edge_feat)
    
    # 转换为张量
    edge_features = torch.stack(edge_features)
    
    return g, node_features, edge_features

def calculate_msa_reward(latency, jitter, cost, weights):
    """
    计算微服务调度智能体的奖励
    奖励是加权目标函数的负值
    
    参数:
        latency: 标准化的应用延迟(0到1)
        jitter: 标准化的延迟抖动(0到1)
        cost: 标准化的公有云成本(0到1)
        weights: (w1, w2, w3)元组，表示延迟、抖动和成本的权重
    
    返回:
        float: 奖励值(加权目标的负值)
    """
    w1, w2, w3 = weights  # 解包权重
    
    # 计算加权目标(越低越好)
    weighted_objective = w1 * latency + w2 * jitter + w3 * cost
    
    # 返回加权目标的负值(奖励越高越好)
    return -weighted_objective