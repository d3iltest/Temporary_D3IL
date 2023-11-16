python run.py --config-name=sorting_6_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=act_agent \
              agent_name=act \
              window_size=3 \
              group=sorting_6_act_seeds \
              simulation.n_cores=30 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=20