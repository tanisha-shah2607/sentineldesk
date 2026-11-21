"""Owner: S5 (bonus). Outage burst detection over a rolling window of recent tickets."""


class BurstDetector:
    def __init__(self, window: int = 20, sim_threshold: float = 0.8, min_cluster: int = 3):
        self.window, self.sim_threshold, self.min_cluster = window, sim_threshold, min_cluster

    def add(self, ticket_text: str) -> dict:
        # TODO(S5): embed, compare with recent, return banner info
        return {"burst": False, "count": 0}
