python run.py --config-name=sorting_2_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ibc_agent \
              agent_name=ibc \
              window_size=1 \
              group=sorting_2_ibc_seeds \
              simulation.n_cores=30 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=8 \
              agents.sampler.sampler_stepsize_init=0.0493