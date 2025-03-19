from easydict import EasyDict

hybrid_cloud_mappo_config = EasyDict(dict(
    exp_name='hybrid_cloud_mappo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=20,
        type='hybrid_cloud',
        import_names=['dizoo.hybridcloud.envs.hybrid_cloud'],
        num_local_servers=5,
        num_remote_servers=5,
        num_microservices=10,
        max_steps=200,
        remote_access_delay=10.0,
        remote_cost_per_cpu=0.1,
        load_file=None,  # Path to workload file, if None, use default workload
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        action_space='hybrid',  # hybrid action space: continuous for WAA, discrete for MSA
        model=dict(
            # Agent 0: WAA (Weight Adjustment Agent)
            agent_0=dict(
                action_space='continuous',
                model_type='continuous',
                share_encoder=True,
                encoder=dict(
                    fc_encoder=dict(
                        input_dim=6,  # [load, local_util, remote_util, latency, jitter, cost]
                        hidden_dim=128,
                        output_dim=128,
                        activation='ReLU',
                    ),
                ),
                actor=dict(
                    head_type='regression',
                    head_hidden_size=64,
                    activation='ReLU',
                ),
                critic=dict(
                    head_hidden_size=64,
                    activation='ReLU',
                ),
            ),
            # Agent 1: MSA (Microservice Scheduling Agent)
            agent_1=dict(
                action_space='discrete',
                model_type='discrete',
                share_encoder=True,
                encoder=dict(
                    fc_encoder=dict(
                        input_dim=10 * 2 + 10 * 10 + 10 + 3,  # ms_features + call_graph + deployment + weights
                        hidden_dim=256,
                        output_dim=256,
                        activation='ReLU',
                    ),
                ),
                actor=dict(
                    head_type='discrete',
                    head_hidden_size=128,
                    activation='ReLU',
                ),
                critic=dict(
                    head_hidden_size=128,
                    activation='ReLU',
                ),
            ),
        ),
        learn=dict(
            epoch_per_collect=5,
            batch_size=64,
            learning_rate=3e-4,
            # GAE lambda
            gae_lambda=0.95,
            # PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # PPO policy value coefficient
            value_norm=True,
            value_weight=0.5,
            # PPO entropy coefficient
            entropy_weight=0.01,
            # Whether to use advantage norm in a whole training batch
            adv_norm=True,
            # Whether to use value norm in a whole training batch
            ignore_done=False,
        ),
        collect=dict(
            n_sample=1024,
            unroll_len=1,
            discount_factor=0.99,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
            ),
        ),
    ),
))

hybrid_cloud_mappo_create_config = EasyDict(dict(
    env=dict(
        type='hybrid_cloud',
        import_names=['dizoo.hybridcloud.envs.hybrid_cloud'],
    ),
    env_manager=dict(
        cfg_type='BaseEnvManagerDict',
    ),
    policy=dict(
        type='ppo',
        import_names=['ding.policy.ppo'],
    ),
))

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c hybrid_cloud_mappo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((hybrid_cloud_mappo_config, hybrid_cloud_mappo_create_config), seed=0)