import os
import torch
import numpy as np
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.utils import set_pkg_seed
from dizoo.hybridcloud.envs.hybrid_cloud import HybridCloudEnv
from dizoo.hybridcloud.config.hybrid_cloud_mappo_config import hybrid_cloud_mappo_config, hybrid_cloud_mappo_create_config


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        save_cfg=True
    )
    
    # Set random seed
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    
    # Create env and policy
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    
    # Create environments
    collector_env = BaseEnvManager(
        env_fn=[lambda: HybridCloudEnv(cfg.env) for _ in range(collector_env_num)],
        cfg=cfg.env.manager
    )
    evaluator_env = BaseEnvManager(
        env_fn=[lambda: HybridCloudEnv(cfg.env) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager
    )
    
    # Set up collector and evaluator
    collector = SampleSerialCollector(
        env=collector_env,
        policy=policy,
        cfg=cfg.policy.collect.collector,
    )
    evaluator = InteractionSerialEvaluator(
        env=evaluator_env,
        policy=policy,
        cfg=cfg.policy.eval.evaluator,
    )
    
    # Set up learner
    learner = BaseLearner(
        cfg.policy.learn.learner,
        policy.learn_mode,
        tb_logger=SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial')),
    )
    
    # Create policy
    policy = PPOPolicy(cfg.policy)
    
    # Start training
    for _ in range(max_iterations):
        # Collect data through collector and store in replay buffer
        new_data = collector.collect(n_sample=cfg.policy.collect.n_sample)
        learner.train(new_data, collector.envstep)
        
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
    
    # Close environments
    collector.close()
    evaluator.close()
    learner.close()


if __name__ == "__main__":
    main(hybrid_cloud_mappo_config, seed=0)