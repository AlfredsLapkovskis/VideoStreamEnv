from stable_baselines3.common.callbacks import BaseCallback


class RLCallback(BaseCallback):

    def __init__(self, agent, verbose=0):
        super().__init__(verbose)

        self.agent = agent
        self._began = False
        self._step = 0


    def _on_rollout_start(self):
        pass
    

    def _on_training_end(self):
        if self._began:
            self._began = False
            self.agent._end_tracking_usage()
            self._step = 0


    def _on_step(self):
        if self._began:
            self.agent._end_tracking_usage()
        else:
            self._began = True
        
        self.agent._begin_tracking_usage()

        return True


    def _on_rollout_end(self):
        pass


    def _on_training_start(self):
        pass