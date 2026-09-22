"""Select and analyze a best-compromise solution from a MOPSO Pareto front.

Reads the pareto_front.csv / pareto_solutions.csv produced by
optimize_stage_1.py, optionally visualizes the front (pairwise 2D
non-dominance plots + a 3D scatter), and picks the best-compromise
solution via a normalized weighted sum over the objectives.

The paper's Case 1-4 configurations correspond to these objective weights:
    Case 1 (coverage-only):    {'coverage_rate': 1, 'compactness': 0, 'separation': 0}
    Case 2 (compactness-only): {'coverage_rate': 0, 'compactness': 1, 'separation': 0}
    Case 3 (separation-only):  {'coverage_rate': 0, 'compactness': 0, 'separation': 1}
    Case 4 (balanced):         {'coverage_rate': 1/3, 'compactness': 1/3, 'separation': 1/3}
"""
import os
import argparse
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CASE_WEIGHTS = {
    'case1': {'coverage_rate': 1.0, 'compactness': 0.0, 'separation': 0.0},
    'case2': {'coverage_rate': 0.0, 'compactness': 1.0, 'separation': 0.0},
    'case3': {'coverage_rate': 0.0, 'compactness': 0.0, 'separation': 1.0},
    'case4': {'coverage_rate': 1 / 3, 'compactness': 1 / 3, 'separation': 1 / 3},
}


def get_best_compromise(repository, weight):
    """Select the solution minimizing a min-max-normalized weighted sum over objectives."""
    objective_names = list(repository[0]['objectives'].keys())
    obj_min = {}
    obj_max = {}
    for obj_name in objective_names:
        values = [particle['objectives'][obj_name] for particle in repository]
        obj_min[obj_name] = min(values)
        obj_max[obj_name] = max(values)

    best_score = float('inf')
    best_solution = None
    for particle in repository:
        weighted_sum = 0.0
        for obj_name in objective_names:
            if abs(obj_max[obj_name] - obj_min[obj_name]) < 1e-10:
                normalized_value = 0.0
            else:
                # Normalize to [0, 1]; lower is better for all objectives.
                normalized_value = (particle['objectives'][obj_name] - obj_min[obj_name]) / (
                        obj_max[obj_name] - obj_min[obj_name])
            weighted_sum += weight.get(obj_name, 1) * normalized_value
        if weighted_sum < best_score:
            best_score = weighted_sum
            best_solution = particle
    return best_solution, best_score


def is_dominated_2d(solution, all_solutions, obj1, obj2):
    """Check whether `solution` is Pareto-dominated when considering only two objectives."""
    for other in all_solutions:
        if other is solution:
            continue
        if (other['objectives'][obj1] <= solution['objectives'][obj1] and
                other['objectives'][obj2] <= solution['objectives'][obj2] and
                (other['objectives'][obj1] < solution['objectives'][obj1] or
                 other['objectives'][obj2] < solution['objectives'][obj2])):
            return True
    return False


def visualize_pareto_front(repository):
    """Plot pairwise 2D non-dominance scatterplots and a 3D scatter of the full Pareto front."""
    objective_names = list(repository[0]['objectives'].keys())

    obj_pairs = list(itertools.combinations(objective_names, 2))
    for obj1, obj2 in obj_pairs:
        plt.figure(figsize=(10, 8))
        for sol in repository:
            dominated_2d = is_dominated_2d(sol, repository, obj1, obj2)
            color = 'blue' if dominated_2d else 'red'
            plt.scatter(sol['objectives'][obj1], sol['objectives'][obj2], color=color, alpha=0.7)
        plt.xlabel(obj1)
        plt.ylabel(obj2)
        plt.title(f'Pareto Front: {obj1} vs {obj2}')
        plt.grid(True)
        plt.scatter([], [], color='red', label='Non-dominated')
        plt.scatter([], [], color='blue', label='Dominated')
        plt.legend()
        plt.show()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for sol in repository:
        ax.scatter(sol['objectives'][objective_names[0]],
                   sol['objectives'][objective_names[1]],
                   sol['objectives'][objective_names[2]],
                   color='red', alpha=0.7)
    ax.set_xlabel(objective_names[0])
    ax.set_ylabel(objective_names[1])
    ax.set_zlabel(objective_names[2])
    ax.set_title('3D Pareto Front')
    plt.grid(True)
    plt.legend(['Non-dominated'], loc='upper right')
    plt.show()


def main(args):
    pareto_front_path = os.path.join(args.input_folder, "pareto_front.csv")
    if not os.path.exists(pareto_front_path):
        print(f"Pareto front file not found: {pareto_front_path}")
        return
    pareto_front_df = pd.read_csv(pareto_front_path)

    pareto_solutions_path = os.path.join(args.input_folder, "pareto_solutions.csv")
    if not os.path.exists(pareto_solutions_path):
        print(f"Parameter values file not found: {pareto_solutions_path}")
        return
    solutions_df = pd.read_csv(pareto_solutions_path)

    repository = []
    for index in range(len(pareto_front_df)):
        objectives = pareto_front_df.iloc[index].to_dict()
        params = solutions_df.iloc[index].to_dict()
        repository.append({'objectives': objectives, 'params': params})

    if args.visualize:
        visualize_pareto_front(repository)

    weights = CASE_WEIGHTS[args.case]
    best_solution, min_weighted_sum = get_best_compromise(repository, weights)
    if best_solution is None:
        print("No best-compromise solution found")
        return

    os.makedirs(args.output_folder, exist_ok=True)
    print(f"=== Best-compromise solution ({args.case}) ===")
    print(f"Weighted sum: {min_weighted_sum}")
    print("Objective values:")
    for obj_name, value in best_solution['objectives'].items():
        print(f"{obj_name}: {value}")
    print("Parameter values:")
    for param_name, value in best_solution['params'].items():
        print(f"{param_name}: {value}")

    best_objective_path = os.path.join(args.output_folder, f"best_objective_{args.case}.csv")
    pd.DataFrame([best_solution['objectives']]).to_csv(best_objective_path, index=False)
    print(f"Saved best-compromise objective values to {best_objective_path}")

    best_solutions_path = os.path.join(args.output_folder, f"best_solution_{args.case}.csv")
    pd.DataFrame([best_solution['params']]).to_csv(best_solutions_path, index=False)
    print(f"Saved best-compromise parameter values to {best_solutions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a best-compromise solution from a MOPSO Pareto front")
    parser.add_argument("--input_folder", default="results/optimize_stage_1",
                         help="Folder containing pareto_front.csv / pareto_solutions.csv (from optimize_stage_1.py)")
    parser.add_argument("--output_folder", default="results/optimize_stage_2", help="Folder to save the selected solution")
    parser.add_argument("--case", choices=list(CASE_WEIGHTS.keys()), default="case4",
                         help="Which objective-weighting preset to use (see module docstring)")
    parser.add_argument('--visualize', action="store_true", default=False, help="Visualize the Pareto front")
    args = parser.parse_args()
    main(args)
