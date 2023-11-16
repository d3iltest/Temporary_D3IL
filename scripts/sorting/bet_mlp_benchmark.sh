python run.py --config-name=sorting_2_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=bet_mlp_agent \
              agent_name=bet_mlp \
              window_size=1 \
              group=sorting_2_bet_mlp_seeds \
              simulation.n_cores=10 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=8 \
              agents.model.vocab_size=24 \
              agents.model.offset_loss_scale=1.0