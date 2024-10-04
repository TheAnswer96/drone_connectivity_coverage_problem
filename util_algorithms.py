import gurobipy as gp
from gurobipy import GRB


def solve_set_cover_OPT(universe, collection):
    # Create a new model
    model = gp.Model("SetCover")
    model.setParam('OutputFlag', False)

    # Create variables: x[j] is 1 if subset j is in the cover, 0 otherwise
    x = model.addVars(len(collection), vtype=GRB.BINARY, name="x")

    # Set objective: minimize the number of subsets in the cover
    model.setObjective(gp.quicksum(x[j] for j in range(len(collection))), GRB.MINIMIZE)

    # Add constraints: each element in the universe must be covered by at least one subset
    for element in universe:
        model.addConstr(gp.quicksum(x[j] for j in range(len(collection)) if element in collection[j]) >= 1,
                        name=f"Cover_{element}")

    # Optimize the model
    model.optimize()

    # Get the result
    selected_subsets = [j for j in range(len(collection)) if x[j].x > 0.5]

    return selected_subsets


def solve_set_cover_APX(universe, collection):
    # APX: greedy algorithm that selects at every step the collection with the maximum coverage until the universe is covered
    uncovered = universe.copy()
    selected_collections_index = []
    selected_collections = []

    # print("Initial Universe:", universe)
    # print("Initial Collection:", collection)

    while uncovered:
        best = None
        best_index = -1
        max_cover = 0
        for index, col in enumerate(collection):
            intersection = uncovered & col
            intersection_size = len(intersection)
            # print(f"Subset {index}: {col}, Covers {intersection_size} uncovered elements")
            if intersection_size > max_cover:
                max_cover = intersection_size
                best = col
                best_index = index
        if max_cover == 0:
            # print("No subset can cover any more uncovered elements.")
            break
        selected_collections_index.append(best_index)
        selected_collections.append(best)
        uncovered = uncovered - best
        # print(f"\nSelected Subset {best_index}: {best}")
        # print(f"Uncovered Elements Remaining: {uncovered}\n")

    # print("Selected Subsets Indices:", selected_collections_index)
    # print("Selected Subsets Collections:", selected_collections)
    return selected_collections
