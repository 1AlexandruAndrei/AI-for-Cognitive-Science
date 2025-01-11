import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt

def generate_3sat_example(num_vars, num_clauses):
    clauses = []
    for _ in range(num_clauses):
        clause = random.sample(range(1, num_vars + 1), 3)
        clause = [var if random.random() > 0.5 else -var for var in clause]
        clauses.append(clause)
    return clauses

def format_3sat_expression(clauses):
    formatted_clauses = []
    for clause in clauses:
        formatted_clause = " ∨ ".join([f"x{abs(literal)}" if literal > 0 else f"¬x{abs(literal)}" for literal in clause])
        formatted_clauses.append(f"({formatted_clause})")
    return " ∧ ".join(formatted_clauses)


def evaluate_clause(clause, solution):
    return any((literal > 0 and solution[abs(literal)-1]) or
               (literal < 0 and not solution[abs(literal)-1]) for literal in clause)


def evaluate_3sat(clauses, solution):
    return all(evaluate_clause(clause, solution) for clause in clauses)


def find_perfect_solution(clauses, num_vars):
    for candidate in itertools.product([True, False], repeat=num_vars):
        if evaluate_3sat(clauses, candidate):
            return candidate
    return None

def plot_3sat_graph(clauses, solution=None, title="3SAT Graph"):
    G = nx.Graph()
    for i, clause in enumerate(clauses):
        clause_name = f"C{i+1}"
        G.add_node(clause_name, color='red')
        for literal in clause:
            var_name = f"x{abs(literal)}"
            G.add_node(var_name, color='blue')
            G.add_edge(clause_name, var_name)


    colors = [data['color'] for _, data in G.nodes(data=True)]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='gray')
    plt.title(title)
    plt.show()


def test_3sat():

    num_vars = 4
    num_clauses = 3
    clauses = generate_3sat_example(num_vars, num_clauses)

    print("Expresia 3SAT generată:")
    print(format_3sat_expression(clauses))

    perfect_solution = find_perfect_solution(clauses, num_vars)
    chance_solution = [random.choice([True, False]) for _ in range(num_vars)]

    print("\nSoluție Perfect AI (toate clauzele satisfăcute):")
    if perfect_solution:
        print(f"Valori variabile: {perfect_solution}")
        for i, clause in enumerate(clauses):
            clause_result = evaluate_clause(clause, perfect_solution)
            print(f"Clauza {i+1}: {clause} -> {'True' if clause_result else 'False'}")
        print(f"Rezultat total: {'Satisfăcut' if evaluate_3sat(clauses, perfect_solution) else 'Nesatisfăcut'}")
    else:
        print("Nu există soluție care să satisfacă toate clauzele.")

    print("\nSoluție Chance AI (aleatoare):")
    print(f"Valori variabile: {chance_solution}")
    for i, clause in enumerate(clauses):
        clause_result = evaluate_clause(clause, chance_solution)
        print(f"Clauza {i+1}: {clause} -> {'True' if clause_result else 'False'}")
    print(f"Rezultat total: {'Satisfăcut' if evaluate_3sat(clauses, chance_solution) else 'Nesatisfăcut'}")

    plot_3sat_graph(clauses, perfect_solution, title="3SAT Visualization")

test_3sat()
