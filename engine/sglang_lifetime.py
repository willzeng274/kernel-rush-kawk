"""Optional-path deadline, published owners, and sticky failed-drain state."""
import time


class CandidateRejected(Exception):
    pass


class PoisonedRuntime(RuntimeError):
    pass


class Control:
    def __init__(self, started, drain, clock=time.monotonic):
        self.started, self._drain, self.clock = started, drain, clock
        self.owners = []
        self.poisoned = False
        self.deadline = started + 270.0
        self.base = self.candidate = self.restore = 0.0
        self.phases = {}

    def register(self, owner):
        self.healthy()
        if not any(x is owner for x in self.owners):
            self.owners.append(owner)
        return owner

    def healthy(self):
        if self.poisoned:
            raise PoisonedRuntime("optional device runtime has a failed drain")

    def drain(self):
        self.wait(self._drain)

    def wait(self, operation):
        self.healthy()
        try:
            operation()
        except BaseException:
            self.poisoned = True
            raise

    def release(self, owner):
        self.drain()
        self.owners[:] = [x for x in self.owners if x is not owner]

    def reserve(self):
        if self.base <= 0 or self.restore <= 0:
            raise CandidateRejected("retained generation and restore measurements required")
        spare = 15.0 + 2 * max(self.base, self.candidate) + 2 * self.restore
        self.deadline = min(self.deadline, self.started + 300.0 - spare)
        return self.deadline

    def live(self, phase=None, floor=0.0):
        self.healthy()
        reserve = max(floor, 2 * self.phases.get(phase, 0.0))
        if self.clock() + reserve >= self.deadline:
            raise TimeoutError("optional setup deadline")

    def observed(self, phase, elapsed):
        self.phases[phase] = max(self.phases.get(phase, 0.0), elapsed)
        if phase == "base":
            self.base = max(self.base, elapsed)
        elif phase == "candidate":
            self.candidate = max(self.candidate, elapsed)
        elif phase == "restore":
            self.restore = max(self.restore, elapsed)
        if self.base and self.restore:
            self.reserve()
        self.live()


class Temporaries:
    """Publish before any asynchronous optional allocation/copy."""
    def __init__(self, control):
        self.control, self.items = control, []
        control.register(self)

    def hold(self, value):
        self.items.append(value)
        return value

    def close(self):
        self.control.release(self)
        self.items.clear()
