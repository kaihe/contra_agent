
# =============================================================================
# CURRICULUM CALLBACK
# =============================================================================

class CurriculumCallback(BaseCallback):
    """Advances the training start state to the agent's frontier every eval_freq steps."""

    def __init__(self, game, state, eval_freq=1_000_000, state_dir="curriculum_states",
                 random_start_frames=0, verbose=0):
        super().__init__(verbose)
        self.game = game
        self.eval_freq = eval_freq
        self.state_dir = state_dir
        self.random_start_frames = random_start_frames
        self.best_distance = 0
        self.current_state = state
        self.steps_since_eval = 0
        os.makedirs(state_dir, exist_ok=True)

    def _on_step(self) -> bool:
        self.steps_since_eval += self.training_env.num_envs
        if self.steps_since_eval >= self.eval_freq:
            self.steps_since_eval = 0
            self._evaluate_and_advance()
        return True

    def _evaluate_and_advance(self):
        # Create single eval env in main process
        eval_base = retro.make(
            game=self.game,
            state=self.current_state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
        )
        eval_env = ContraWrapper(eval_base)

        obs, info = eval_env.reset()
        best_snapshot = None
        best_dist = 0

        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = eval_env.step(action)
            done = term or trunc

            curr_dist = eval_env.max_x_reached
            if curr_dist > best_dist:
                best_dist = curr_dist
                best_snapshot = eval_env.unwrapped.em.get_state()

        eval_env.close()

        if best_dist <= 0 or best_snapshot is None:
            print(f"[Curriculum] No progress in eval (best_dist={best_dist}). Skipping.")
            return

        total_dist = self.best_distance + best_dist

        # Set lives=2 in the snapshot
        tmp_env = retro.make(
            game=self.game,
            state=self.current_state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode=None,
        )
        tmp_env.reset()
        tmp_env.em.set_state(best_snapshot)
        tmp_env.data.update_ram()
        tmp_env.data.set_value("lives", 2)
        tmp_env.data.update_ram()
        final_state = tmp_env.em.get_state()
        tmp_env.close()

        # Save state file
        state_name = f"curriculum_dist{total_dist}"
        state_path = os.path.join(self.state_dir, f"{state_name}.state")
        with gzip.open(state_path, "wb") as f:
            f.write(final_state)

        # Copy to retro game data dir so retro.make(state=name) works
        game_data_dir = retro.data.get_game_path(self.game)
        shutil.copy(state_path, os.path.join(game_data_dir, f"{state_name}.state"))

        self.best_distance = total_dist
        self.current_state = state_name

        print(f"[Curriculum] New frontier: distance={total_dist}, state={state_name}")

        # Recreate training envs with new state
        new_env = SubprocVecEnv(
            [make_env(self.game, state=state_name, seed=i,
                      random_start_frames=self.random_start_frames)
             for i in range(NUM_ENV)]
        )
        self.model.set_env(new_env)