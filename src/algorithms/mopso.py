import numpy as np
import random
from tqdm import tqdm


class Particle:
    """A single particle in the MOPSO swarm."""

    def __init__(self, position, velocity, bounds):
        self.position = position.copy()  # Current position (parameter values).
        self.velocity = velocity.copy()
        self.personal_best = position.copy()
        self.objectives = None
        self.dominated = False
        self.crowding_distance = 0.0
        self.bounds = bounds

    def update_velocity(self, global_best, w=0.7, c1=1.5, c2=1.5):
        """Update velocity using cognitive (personal-best) and social (leader) components."""
        r1 = np.random.random(len(self.position))
        r2 = np.random.random(len(self.position))
        cognitive = c1 * r1 * (self.personal_best - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        """Advance the particle and clip it back into the parameter bounds."""
        self.position = self.position + self.velocity
        for i in range(len(self.position)):
            low, high = self.bounds[i]
            self.position[i] = max(low, min(high, self.position[i]))


class MOPSO:
    """Multi-Objective Particle Swarm Optimization.

    Maintains an archive ("repository") of non-dominated solutions found so
    far, using crowding-distance-based tournament selection to pick a leader
    for each particle -- used to tune MulDet3D's clustering parameters
    (alpha, rho_min) against the coverage/compactness/separation objectives
    (see paper Sec. "Multi-Objective Parameter Optimization").
    """

    def __init__(self, obj_func, param_bounds, obj_func_params, obj_name,
                 swarm_size=30, max_iter=100):
        self.obj_func = obj_func  # Multi-objective function.
        self.obj_name = obj_name  # Objective names.
        self.obj_num = len(obj_name)
        self.param_bounds = param_bounds
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.repository = []  # Non-dominated solution archive.
        self.max_repository_size = 2 * self.swarm_size
        self.swarm = self._initialize_swarm()
        self.obj_func_params = obj_func_params  # Extra arguments passed through to obj_func.

    def _initialize_swarm(self):
        """Randomly initialize particle positions and velocities within the parameter bounds."""
        swarm = []
        dim = len(self.param_bounds)
        for _ in range(self.swarm_size):
            position = np.array([np.random.uniform(low, high) for low, high in self.param_bounds])
            velocity = np.array([np.random.uniform(-0.1, 0.1) for _ in range(dim)])
            particle = Particle(position, velocity, self.param_bounds)
            swarm.append(particle)
        return swarm

    def _evaluate_swarm(self):
        """Evaluate the objective function for every particle in the swarm."""
        for particle in tqdm(self.swarm):
            particle.objectives = self.obj_func(particle.position, self.obj_func_params)

    def _update_personal_best(self):
        """Update each particle's personal-best position, reusing already-computed objectives."""
        for particle in self.swarm:
            current_objectives = particle.objectives
            if not hasattr(particle, 'personal_best_objectives'):
                particle.personal_best_objectives = self.obj_func(particle.personal_best, self.obj_func_params)
            if self._dominates(current_objectives, particle.personal_best_objectives) or \
                    not self._dominates(particle.personal_best_objectives, current_objectives):
                particle.personal_best = particle.position.copy()
                particle.personal_best_objectives = current_objectives

    def _dominates(self, obj1, obj2):
        """Return True if obj1 Pareto-dominates obj2 (minimization, all objectives <=, one <)."""
        at_least_one_better = False
        for i in range(self.obj_num):
            if obj1[i] > obj2[i]:
                return False
            elif obj1[i] < obj2[i]:
                at_least_one_better = True
        return at_least_one_better

    def _update_repository(self):
        """Update the non-dominated solution archive with the current swarm."""
        for particle in self.swarm:
            dominated = False
            i = 0
            while i < len(self.repository):
                repo_particle = self.repository[i]
                if self._dominates(repo_particle.objectives, particle.objectives):
                    dominated = True
                    break
                elif self._dominates(particle.objectives, repo_particle.objectives):
                    self.repository.pop(i)  # This particle dominates an archived solution; remove it.
                else:
                    i += 1
            if not dominated:
                is_duplicate = any(
                    np.array_equal(repo_particle.position, particle.position) for repo_particle in self.repository)
                if not is_duplicate:
                    self.repository.append(Particle(particle.position, particle.velocity, self.param_bounds))
                    self.repository[-1].objectives = particle.objectives

        if len(self.repository) > self.max_repository_size:
            self._compute_crowding_distance()
            self.repository.sort(key=lambda p: p.crowding_distance, reverse=True)
            self.repository = self.repository[:self.max_repository_size]

    def _compute_crowding_distance(self):
        """Compute NSGA-II-style crowding distance for each archived solution."""
        n = len(self.repository)
        if n <= 2:
            for particle in self.repository:
                particle.crowding_distance = float('inf')
            return
        for obj in range(self.obj_num):
            self.repository.sort(key=lambda p: p.objectives[obj])
            self.repository[0].crowding_distance = float('inf')
            self.repository[-1].crowding_distance = float('inf')
            obj_max = self.repository[-1].objectives[obj]
            obj_min = self.repository[0].objectives[obj]
            scale = max(1e-10, obj_max - obj_min)
            for i in range(1, n - 1):
                distance = (self.repository[i + 1].objectives[obj] - self.repository[i - 1].objectives[obj]) / scale
                self.repository[i].crowding_distance += distance

    def _select_leader(self):
        """Pick a leader via binary crowding-distance tournament over the archive."""
        if not self.repository:
            return np.array([np.random.uniform(low, high) for low, high in self.param_bounds])
        if len(self.repository) == 1:
            return self.repository[0].position
        candidates = random.sample(self.repository, min(2, len(self.repository)))
        if candidates[0].crowding_distance > candidates[1].crowding_distance:
            return candidates[0].position
        else:
            return candidates[1].position

    def optimize(self):
        """Run the full MOPSO loop and return the final Pareto archive."""
        print("Evaluating initial swarm...")
        self._evaluate_swarm()
        print("Updating repository...")
        self._update_repository()
        print(f"Initial non-dominated solutions: {len(self.repository)}")
        for iteration in tqdm(range(self.max_iter)):
            print(f"MOPSO iteration {iteration + 1}/{self.max_iter}")
            for particle in self.swarm:
                leader = self._select_leader()
                particle.update_velocity(leader)
                particle.update_position()
            self._evaluate_swarm()
            self._update_personal_best()
            self._update_repository()
            if len(self.repository) > 0:
                print(f"Current repository size: {len(self.repository)}")
                print("Current minimum objective values:")
                for obj, key in enumerate(self.obj_name):
                    min_value = min(particle.objectives[obj] for particle in self.repository)
                    print(f"{key}: {min_value}")
        return self.repository
