class HybridCloudScheduler:
    def __init__(self, waa_model_path, msa_model_path, waa_state_dim=6, 
                node_feat_dim=6, edge_feat_dim=3, num_placement_options=2):
        """
        初始化混合云调度器，加载预训练模型
        
        参数:
            waa_model_path: WAA模型路径
            msa_model_path: MSA模型路径
            waa_state_dim: WAA状态维度
            node_feat_dim: 节点特征维度
            edge_feat_dim: 边特征维度
            num_placement_options: 部署选项数量
        """
        # 初始化模型
        self.waa = WeightAdjustmentAgent(waa_state_dim)
        self.msa = MicroserviceSchedulingAgent(node_feat_dim, edge_feat_dim, 
                                             num_placement_options=num_placement_options)
        
        # 加载预训练模型
        self.waa.load_state_dict(torch.load(waa_model_path))
        self.msa.load_state_dict(torch.load(msa_model_path))
        
        # 设置为评估模式
        self.waa.eval()
        self.msa.eval()
        
        # 跟踪当前权重
        self.current_weights = None
        
        print("混合云微服务调度系统已初始化")
    
    def update_system_metrics(self, system_metrics):
        """
        根据当前系统指标更新目标权重
        
        参数:
            system_metrics: 系统指标字典
        
        返回:
            numpy.ndarray: 更新的权重(w1, w2, w3)
        """
        with torch.no_grad():
            # 构建WAA状态
            waa_state = build_waa_state(system_metrics)
            
            # 获取更新的权重
            weights = self.waa.get_weights(waa_state)
            self.current_weights = weights.numpy()
            
            print(f"目标权重已更新: 延迟={self.current_weights[0]:.2f}, "
                  f"抖动={self.current_weights[1]:.2f}, 成本={self.current_weights[2]:.2f}")
            
            return self.current_weights
    
    def schedule_microservices(self, microservices, dependencies):
        """
        为微服务生成调度决策
        
        参数:
            microservices: 包含属性的微服务容器列表
            dependencies: 微服务间的依赖关系列表
        
        返回:
            list: 每个微服务的部署决策
        """
        if self.current_weights is None:
            raise ValueError("必须先更新系统指标以确定权重")
        
        with torch.no_grad():
            # 构建微服务图
            g, node_features, edge_features = build_microservice_graph(
                microservices, dependencies)
            
            # 将权重转换为张量
            weights = torch.tensor(self.current_weights, dtype=torch.float32)
            
            # 获取部署决策
            action_probs, _ = self.msa(g, node_features, edge_features, weights)
            
            # 转换为部署决策
            # action = 0: 部署在私有云中
            # action = 1: 部署在公有云中
            placements = action_probs.argmax(dim=1).numpy()
            
            return placements
    
    def generate_schedule(self, system_state):
        """
        根据当前系统状态生成完整调度方案
        
        参数:
            system_state: 包含system_metrics、microservices和dependencies的字典
        
        返回:
            dict: 包含权重和部署的调度计划
        """
        # 根据系统指标更新权重
        weights = self.update_system_metrics(system_state['system_metrics'])
        
        # 生成部署决策
        placements = self.schedule_microservices(
            system_state['microservices'], system_state['dependencies'])
        
        # 创建调度计划
        schedule = {
            'weights': {
                'latency_weight': float(weights[0]),
                'jitter_weight': float(weights[1]),
                'cost_weight': float(weights[2])
            },
            'placements': []
        }
        
        # 添加详细部署信息
        for i, ms in enumerate(system_state['microservices']):
            placement = {
                'microservice_id': ms['id'],
                'microservice_name': ms['name'],
                'target': 'private_cloud' if placements[i] == 0 else 'public_cloud',
                'confidence': float(action_probs[i][placements[i]].item())
            }
            schedule['placements'].append(placement)
        
        # 计算部署摘要
        private_count = sum(1 for p in schedule['placements'] if p['target'] == 'private_cloud')
        public_count = sum(1 for p in schedule['placements'] if p['target'] == 'public_cloud')
        
        print(f"调度方案已生成: {private_count}个微服务部署在私有云, {public_count}个在公有云")
        
        return schedule

def system_monitoring_loop(scheduler, monitoring_interval=60):
    """
    主系统监控和调度循环
    
    参数:
        scheduler: HybridCloudScheduler实例
        monitoring_interval: 监控周期间隔(秒)
    """
    import time
    
    print("启动混合云微服务调度系统...")
    
    while True:
        try:
            # 1. 收集当前系统状态
            system_state = collect_system_state()
            
            # 2. 生成调度方案
            schedule = scheduler.generate_schedule(system_state)
            
            # 3. 应用调度方案
            apply_schedule(schedule)
            
            # 4. 记录结果
            log_schedule_results(schedule, system_state)
            
            # 等待下一个监控周期
            print(f"等待{monitoring_interval}秒进行下一次调度...")
            time.sleep(monitoring_interval)
            
        except Exception as e:
            print(f"监控循环中出错: {e}")
            time.sleep(10)  # 等待并重试

def collect_system_state():
    """收集当前系统状态，从监控系统获取"""
    # 在生产环境中，这会从真实的监控工具收集数据
    # 这里提供一个模拟实现
    import random
    
    # 模拟系统指标
    load_intensity = random.uniform(0.2, 0.9)
    system_metrics = {
        'load_intensity': load_intensity,
        'private_cloud_utilization': min(0.4 + load_intensity, 0.95),
        'public_cloud_cost': 0 if load_intensity < 0.6 else random.uniform(0.1, 0.8),
        'avg_latency': 0.1 + load_intensity * 0.5,
        'avg_jitter': 0.05 + load_intensity * 0.3,
        'sla_violation_rate': 0 if load_intensity < 0.7 else random.uniform(0, 0.1)
    }
    
    # 模拟微服务
    microservices = []
    num_services = random.randint(8, 15)
    
    for i in range(num_services):
        microservice = {
            'id': i,
            'name': f'service-{i}',
            'cpu_requirement': random.uniform(0.1, 0.8),
            'memory_requirement': random.uniform(0.1, 0.9),
            'network_requirement': random.uniform(0.05, 0.7),
            'current_load': load_intensity * random.uniform(0.8, 1.2),
            'current_latency': 0.05 + load_intensity * random.uniform(0.1, 0.4),
            'current_jitter': 0.01 + load_intensity * random.uniform(0.05, 0.2)
        }
        microservices.append(microservice)
    
    # 模拟依赖关系
    dependencies = []
    
    # 创建一个连通的服务图
    for i in range(num_services):
        # 每个服务调用1-3个其他服务
        num_calls = random.randint(1, min(3, num_services-1))
        targets = random.sample([j for j in range(num_services) if j != i], num_calls)
        
        for target in targets:
            dependency = {
                'source': i,
                'target': target,
                'call_frequency': random.uniform(0.1, 1.0),
                'data_volume': random.uniform(0.1, 0.9),
                'current_latency': 0.01 + load_intensity * random.uniform(0.05, 0.2)
            }
            dependencies.append(dependency)
    
    print(f"收集系统状态: 负载强度={load_intensity:.2f}, 微服务数量={num_services}")
    
    return {
        'system_metrics': system_metrics,
        'microservices': microservices,
        'dependencies': dependencies
    }

def apply_schedule(schedule):
    """
    应用生成的调度方案到系统中
    在生产环境中，这将调用容器编排系统的API
    
    参数:
        schedule: 调度计划
    """
    # 在生产环境中，这将调用API来迁移容器
    print("\n应用新的调度方案:")
    print(f"目标权重: 延迟={schedule['weights']['latency_weight']:.2f}, "
          f"抖动={schedule['weights']['jitter_weight']:.2f}, "
          f"成本={schedule['weights']['cost_weight']:.2f}")
    
    private_count = sum(1 for p in schedule['placements'] if p['target'] == 'private_cloud')
    public_count = sum(1 for p in schedule['placements'] if p['target'] == 'public_cloud')
    
    print(f"部署 {private_count} 个微服务在私有云, {public_count} 个在公有云")
    
    # 模拟执行调度决策
    for placement in schedule['placements']:
        print(f"  将微服务 {placement['microservice_name']} 部署到 {placement['target']} "
              f"(置信度: {placement['confidence']:.2f})")
    
    # 在实际系统中，我们会在这里调用容器编排API

def log_schedule_results(schedule, system_state):
    """
    记录应用调度方案的结果
    
    参数:
        schedule: 应用的调度方案
        system_state: 系统状态
    """
    import time
    
    # 在生产环境中，这将写入日志系统
    print(f"\n调度方案应用时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 模拟计算目标函数值
    load = system_state['system_metrics']['load_intensity']
    w1 = schedule['weights']['latency_weight']
    w2 = schedule['weights']['jitter_weight']
    w3 = schedule['weights']['cost_weight']
    
    private_count = sum(1 for p in schedule['placements'] if p['target'] == 'private_cloud')
    public_count = sum(1 for p in schedule['placements'] if p['target'] == 'public_cloud')
    
    # 模拟延迟、抖动和成本
    latency = 0.1 + load * (0.3 - 0.2 * public_count / (private_count + public_count + 0.001))
    jitter = 0.05 + load * (0.2 - 0.15 * public_count / (private_count + public_count + 0.001))
    cost = 0.5 * public_count / (private_count + public_count + 0.001)
    
    # 计算加权目标
    weighted_obj = w1 * latency + w2 * jitter + w3 * cost
    
    print(f"性能指标: 延迟={latency:.2f}, 抖动={jitter:.2f}, 成本={cost:.2f}")
    print(f"加权目标值: {weighted_obj:.4f}")
    
    # 在实际系统中，我们会记录详细的性能指标

# 系统启动入口
if __name__ == "__main__":
    # 预训练模型路径
    waa_model_path = "models/waa_model_final.pth"
    msa_model_path = "models/msa_model_final.pth"
    
    # 初始化调度器
    scheduler = HybridCloudScheduler(waa_model_path, msa_model_path)
    
    # 启动监控循环
    system_monitoring_loop(scheduler, monitoring_interval=60)