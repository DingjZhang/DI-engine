import os
import numpy as np
import gymnasium as gym
from typing import Any, Dict, List, Tuple, Union, Optional
from collections import defaultdict

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('hybrid_cloud')
class HybridCloudEnv(BaseEnv):
    """
    Overview:
        Hybrid Cloud Environment for microservice scheduling using multi-agent reinforcement learning.
        This environment simulates a hybrid cloud setup with local and remote server clusters,
        microservices with dependencies, and dynamic workload.
        
    Features:
        - Local server cluster with limited CPU resources
        - Remote server cluster with unlimited CPU resources but with access delay and usage cost
        - Microservices with dependencies forming a call graph
        - Dynamic workload simulation with ability to read from file
        - Two agents: Weight Adjustment Agent (WAA) and Microservice Scheduling Agent (MSA)
    """

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the Hybrid Cloud Environment.
        Arguments:
            - cfg (:obj:`dict`): Environment configuration.
        """
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        
        # Environment parameters
        self._num_local_servers = cfg.get('num_local_servers', 5)
        self._num_remote_servers = cfg.get('num_remote_servers', 5)
        self._num_microservices = cfg.get('num_microservices', 10)
        self._max_steps = cfg.get('max_steps', 1000)
        self._remote_access_delay = cfg.get('remote_access_delay', 10.0)  # ms
        self._remote_cost_per_cpu = cfg.get('remote_cost_per_cpu', 0.1)  # cost per CPU unit
        self._load_file = cfg.get('load_file', None)
        
        # Multi-agent setup
        self._num_agents = 2  # WAA and MSA
        
        # Action and observation spaces will be initialized in reset()

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Overview:
            Reset the environment to an initial state.
        Returns:
            - obs (:obj:`Dict[str, np.ndarray]`): Initial observations for each agent.
        """
        if not self._init_flag:
            # Initialize action and observation spaces
            self._setup_spaces()
            
            # Load workload data if provided
            if self._load_file and os.path.exists(self._load_file):
                self._load_workload(self._load_file)
            else:
                # Default workload pattern if no file is provided
                self._workload = np.ones(self._max_steps) * 100  # Default 100 requests/s
            
            self._init_flag = True
        
        # Reset environment state
        self._step_count = 0
        self._current_load = self._workload[0]
        
        # Initialize servers
        self._local_servers = [{'cpu_capacity': 100, 'cpu_used': 0} for _ in range(self._num_local_servers)]
        self._remote_servers = [{'cpu_capacity': float('inf'), 'cpu_used': 0} for _ in range(self._num_remote_servers)]
        
        # Initialize microservices with random resource requirements
        self._microservices = self._init_microservices()
        
        # Initialize microservice deployment (all on local servers initially)
        self._deployment = {ms['id']: {'server_id': i % self._num_local_servers, 'is_remote': False} 
                           for i, ms in enumerate(self._microservices)}
        
        # Initialize performance metrics
        self._latency = 0.0
        self._jitter = 0.0
        self._cost = 0.0
        
        # Initialize weights for objectives
        self._weights = np.array([0.33, 0.33, 0.34])  # w1 (latency), w2 (jitter), w3 (cost)
        
        # Calculate initial resource allocation
        self._update_resource_allocation()
        
        # Calculate initial performance metrics
        self._calculate_performance_metrics()
        
        # Reset episode return
        self._eval_episode_return = {agent_id: 0.0 for agent_id in range(self._num_agents)}
        
        # Get initial observations
        obs = self._get_observations()
        
        return obs

    def _setup_spaces(self) -> None:
        """
        Setup action and observation spaces for both agents.
        """
        # WAA (Agent 0) - Continuous action space for weights
        self._waa_action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # MSA (Agent 1) - Discrete action space for microservice placement
        # For each microservice, choose a server (local or remote)
        total_servers = self._num_local_servers + self._num_remote_servers
        self._msa_action_space = gym.spaces.MultiDiscrete(
            [total_servers] * self._num_microservices
        )
        
        # WAA observation space
        self._waa_obs_space = gym.spaces.Box(
            low=0.0, high=float('inf'),
            shape=(6,),  # [load, local_util, remote_util, latency, jitter, cost]
            dtype=np.float32
        )
        
        # MSA observation space
        # Includes microservice features, call graph, current deployment, and weights
        ms_features_dim = 2  # CPU requirement, request rate
        call_graph_dim = self._num_microservices * self._num_microservices  # Adjacency matrix
        deployment_dim = self._num_microservices  # Current server for each microservice
        weights_dim = 3  # Current weights from WAA
        
        self._msa_obs_space = gym.spaces.Box(
            low=0.0, high=float('inf'),
            shape=(ms_features_dim * self._num_microservices + call_graph_dim + deployment_dim + weights_dim,),
            dtype=np.float32
        )
        
        # Combined action and observation spaces
        self._action_space = gym.spaces.Dict({
            0: self._waa_action_space,  # WAA
            1: self._msa_action_space,  # MSA
        })
        
        self._observation_space = gym.spaces.Dict({
            0: self._waa_obs_space,  # WAA
            1: self._msa_obs_space,  # MSA
        })

    def _init_microservices(self) -> List[Dict]:
        """
        Initialize microservices with random resource requirements and call dependencies.
        """
        microservices = []
        for i in range(self._num_microservices):
            # Random CPU requirement between 10 and 50
            cpu_req = np.random.uniform(10, 50)
            
            microservices.append({
                'id': i,
                'name': f'service-{i}',
                'cpu_requirement': cpu_req,
                'request_rate': 0.0,  # Will be updated based on load
            })
        
        # Generate random call graph (directed adjacency matrix)
        call_graph = np.zeros((self._num_microservices, self._num_microservices))
        
        # Ensure each service calls at least one other service (except the last one)
        for i in range(self._num_microservices - 1):
            # Each service can call between 1 and 3 other services
            num_calls = np.random.randint(1, min(4, self._num_microservices - i))
            targets = np.random.choice(
                range(i + 1, self._num_microservices),
                size=num_calls,
                replace=False
            )
            for target in targets:
                # Random call rate between 0.5 and 1.0 (calls per request)
                call_graph[i, target] = np.random.uniform(0.5, 1.0)
        
        self._call_graph = call_graph
        return microservices

    def _load_workload(self, load_file: str) -> None:
        """
        Load workload data from file.
        """
        try:
            with open(load_file, 'r') as f:
                self._workload = np.array([float(line.strip()) for line in f.readlines()])
            
            # If workload is shorter than max_steps, repeat it
            if len(self._workload) < self._max_steps:
                repetitions = int(np.ceil(self._max_steps / len(self._workload)))
                self._workload = np.tile(self._workload, repetitions)[:self._max_steps]
        except Exception as e:
            print(f"Error loading workload file: {e}")
            # Fallback to default workload
            self._workload = np.ones(self._max_steps) * 100

    def _update_resource_allocation(self) -> None:
        """
        Update resource allocation based on current deployment.
        """
        # Reset server resource usage
        for server in self._local_servers:
            server['cpu_used'] = 0
        for server in self._remote_servers:
            server['cpu_used'] = 0
        
        # Allocate resources based on deployment
        for ms in self._microservices:
            ms_id = ms['id']
            deployment_info = self._deployment[ms_id]
            server_id = deployment_info['server_id']
            is_remote = deployment_info['is_remote']
            
            # Update CPU usage
            if is_remote:
                self._remote_servers[server_id]['cpu_used'] += ms['cpu_requirement']
            else:
                self._local_servers[server_id]['cpu_used'] += ms['cpu_requirement']

    def _calculate_performance_metrics(self) -> None:
        """
        Calculate performance metrics (latency, jitter, cost) based on current deployment and load.
        """
        # Update request rates based on current load and call graph
        self._update_request_rates()
        
        # Calculate latency
        latency = 0.0
        for ms in self._microservices:
            ms_id = ms['id']
            deployment_info = self._deployment[ms_id]
            server_id = deployment_info['server_id']
            is_remote = deployment_info['is_remote']
            
            # Base latency depends on CPU allocation and load
            if is_remote:
                server = self._remote_servers[server_id]
                # Remote servers have additional access delay
                base_latency = self._remote_access_delay
            else:
                server = self._local_servers[server_id]
                base_latency = 0.0
            
            # Higher CPU utilization increases latency
            utilization = server['cpu_used'] / server['cpu_capacity'] if server['cpu_capacity'] != float('inf') else 0.0
            load_factor = ms['request_rate'] / ms['cpu_requirement']
            
            # Latency model: base + utilization effect + load effect
            ms_latency = base_latency + 5.0 * utilization + 2.0 * load_factor
            latency += ms_latency * ms['request_rate']
        
        # Normalize latency by total request rate
        total_request_rate = sum(ms['request_rate'] for ms in self._microservices)
        if total_request_rate > 0:
            latency /= total_request_rate
        
        # Calculate jitter (simplified as variance in latency between steps)
        if hasattr(self, '_prev_latency'):
            self._jitter = abs(latency - self._prev_latency)
        else:
            self._jitter = 0.0
        self._prev_latency = latency
        
        # Calculate cost (only for remote servers)
        cost = 0.0
        for server_id in range(self._num_remote_servers):
            cost += self._remote_servers[server_id]['cpu_used'] * self._remote_cost_per_cpu
        
        self._latency = latency
        self._cost = cost
        
        # Apply constraints - if local servers are overloaded, increase latency significantly
        for server in self._local_servers:
            if server['cpu_used'] > server['cpu_capacity']:
                # Penalty for overloading local servers
                overload_ratio = server['cpu_used'] / server['cpu_capacity']
                self._latency *= (1.0 + (overload_ratio - 1.0) * 2.0)

    def _update_request_rates(self) -> None:
        """
        Update request rates for each microservice based on current load and call graph.
        """
        # Initialize request rates (entry point services get the full load)
        entry_points = [i for i in range(self._num_microservices) if not any(self._call_graph[:, i] > 0)]
        if not entry_points:  # If no clear entry points, use the first service
            entry_points = [0]
        
        # Distribute load among entry points
        for ms_id in range(self._num_microservices):
            if ms_id in entry_points:
                self._microservices[ms_id]['request_rate'] = self._current_load / len(entry_points)
            else:
                self._microservices[ms_id]['request_rate'] = 0.0
        
        # Propagate requests through the call graph
        for caller_id in range(self._num_microservices):
            caller_rate = self._microservices[caller_id]['request_rate']
            for callee_id in range(self._num_microservices):
                if self._call_graph[caller_id, callee_id] > 0:
                    call_rate = caller_rate * self._call_graph[caller_id, callee_id]
                    self._microservices[callee_id]['request_rate'] += call_rate

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get observations for both agents.
        """
        # WAA observation
        local_util = sum(server['cpu_used'] for server in self._local_servers) / \
                    sum(server['cpu_capacity'] for server in self._local_servers)
        remote_util = sum(server['cpu_used'] for server in self._remote_servers) / \
                     (self._num_remote_servers * 100)  # Normalize by assuming 100 CPU units per server
        
        waa_obs = np.array([
            self._current_load / 1000,  # Normalize load
            local_util,
            remote_util,
            self._latency / 100,  # Normalize latency
            self._jitter / 10,   # Normalize jitter
            self._cost / 100     # Normalize cost
        ], dtype=np.float32)
        
        # MSA observation
        # Microservice features
        ms_features = []
        for ms in self._microservices:
            ms_features.extend([ms['cpu_requirement'] / 100, ms['request_rate'] / 1000])
        
        # Call graph (flattened adjacency matrix)
        call_graph_flat = self._call_graph.flatten()
        
        # Current deployment
        deployment_vector = np.zeros(self._num_microservices, dtype=np.float32)
        for ms_id, deploy_info in self._deployment.items():
            # Encode server ID and whether it's remote
            server_id = deploy_info['server_id']
            is_remote = deploy_info['is_remote']
            deployment_vector[ms_id] = server_id + (self._num_local_servers if is_remote else 0)
        
        # Combine all features
        msa_obs = np.concatenate([
            np.array(ms_features, dtype=np.float32),
            call_graph_flat,
            deployment_vector,
            self._weights  # Current weights from WAA
        ])
        
        return {
            0: waa_obs,  # WAA observation
            1: msa_obs   # MSA observation
        }
    
    def close(self) -> None:
        """
        Overview:
            Close the environment and release resources.
        """
        self._init_flag = False
    
    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Overview:
            Set random seed for the environment.
        Arguments:
            - seed (:obj:`int`): Random seed.
            - dynamic_seed (:obj:`bool`): Whether to use dynamic seed.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
    
    def random_action(self) -> Dict[int, np.ndarray]:
        """
        Overview:
            Generate random actions for both agents.
        Returns:
            - random_action (:obj:`Dict[int, np.ndarray]`): Random actions for both agents.
        """
        # Random action for WAA (weights)
        waa_action = np.random.random(3)
        waa_action = waa_action / np.sum(waa_action)  # Normalize to sum to 1
        
        # Random action for MSA (microservice placement)
        total_servers = self._num_local_servers + self._num_remote_servers
        msa_action = np.random.randint(0, total_servers, size=self._num_microservices)
        
        return {
            0: waa_action.astype(np.float32),
            1: msa_action.astype(np.int64)
        }
    
    def __repr__(self) -> str:
        """
        Overview:
            Return string representation of the environment.
        Returns:
            - repr (:obj:`str`): String representation.
        """
        return "DI-engine Hybrid Cloud Environment"
    
    @property
    def observation_space(self) -> gym.spaces.Dict:
        """
        Overview:
            Get observation space of the environment.
        Returns:
            - observation_space (:obj:`gym.spaces.Dict`): Observation space.
        """
        return self._observation_space
    
    @property
    def action_space(self) -> gym.spaces.Dict:
        """
        Overview:
            Get action space of the environment.
        Returns:
            - action_space (:obj:`gym.spaces.Dict`): Action space.
        """
        return self._action_space
    
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Create collector environment configuration.
        Arguments:
            - cfg (:obj:`dict`): Original configuration.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of environment configurations.
        """
        collector_env_num = cfg.pop('collector_env_num', 1)
        cfg = {**cfg}
        return [cfg for _ in range(collector_env_num)]
    
    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Create evaluator environment configuration.
        Arguments:
            - cfg (:obj:`dict`): Original configuration.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of environment configurations.
        """
        evaluator_env_num = cfg.pop('evaluator_env_num', 1)
        cfg = {**cfg}
        return [cfg for _ in range(evaluator_env_num)]
    
    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        """
        Overview:
            Enable saving replay data.
        Arguments:
            - replay_path (:obj:`Optional[str]`): Path to save replay data.
        """
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path
        
    def step(self, action: Dict[int, np.ndarray]) -> BaseEnvTimestep:
        """
        Overview:
            Step the environment with actions from both agents.
        Arguments:
            - action (:obj:`Dict[int, np.ndarray]`): Actions from both agents.
                - 0: WAA action (weights for objectives)
                - 1: MSA action (microservice placement)
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): Environment timestep with observation, reward, done, and info.
        """
        # Process WAA action (weights adjustment)
        waa_action = action[0]
        # Normalize weights to sum to 1
        self._weights = waa_action / np.sum(waa_action)
        
        # Process MSA action (microservice placement)
        msa_action = action[1]
        for ms_id, server_choice in enumerate(msa_action):
            # Determine if the server is local or remote
            is_remote = server_choice >= self._num_local_servers
            server_id = server_choice % self._num_local_servers if not is_remote else server_choice - self._num_local_servers
            
            # Update deployment
            self._deployment[ms_id] = {
                'server_id': server_id,
                'is_remote': is_remote
            }
        
        # Update resource allocation based on new deployment
        self._update_resource_allocation()
        
        # Move to next step (update workload)
        self._step_count += 1
        if self._step_count < self._max_steps:
            self._current_load = self._workload[self._step_count]
        
        # Calculate new performance metrics
        prev_latency, prev_jitter, prev_cost = self._latency, self._jitter, self._cost
        self._calculate_performance_metrics()
        
        # Calculate rewards
        # WAA reward: Based on how well the weights match the current system state
        load_threshold = 0.7  # Threshold to determine high/low load
        normalized_load = self._current_load / 1000  # Normalize to 0-1 range
        
        if normalized_load < load_threshold:
            # Low load: prioritize cost (w3)
            waa_reward = self._weights[2] / (self._weights[0] + self._weights[1] + 1e-6)
        else:
            # High load: prioritize performance (w1, w2)
            waa_reward = (self._weights[0] + self._weights[1]) / (self._weights[2] + 1e-6)
        
        # MSA reward: Negative weighted objective value
        weighted_objective = -(self._weights[0] * self._latency + 
                              self._weights[1] * self._jitter + 
                              self._weights[2] * self._cost)
        msa_reward = weighted_objective
        
        # Combined reward (can be adjusted based on specific requirements)
        reward = (waa_reward + msa_reward) / 2
        reward = np.array([reward], dtype=np.float32)
        
        # Update episode return
        self._eval_episode_return[0] += waa_reward
        self._eval_episode_return[1] += msa_reward
        
        # Check if episode is done
        done = self._step_count >= self._max_steps
        
        # Get new observations
        obs = self._get_observations()
        
        # Prepare info
        info = {
            'weights': self._weights,
            'latency': self._latency,
            'jitter': self._jitter,
            'cost': self._cost,
            'load': self._current_load,
            'step': self._step_count,
        }
        
        if done:
            info['eval_episode_return'] = self._eval_episode_return
        
        return BaseEnvTimestep(obs, reward, done, info)