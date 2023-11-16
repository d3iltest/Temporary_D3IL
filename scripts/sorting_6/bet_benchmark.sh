python run.py --config-name=sorting_6_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=bet_agent \
              agent_name=bet \
              window_size=5 \
              group=sorting_6_bet_seeds \
              simulation.n_cores=60 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=20 \
              agents.model.vocab_size=64 \
              agents.model.offset_loss_scale=1.0