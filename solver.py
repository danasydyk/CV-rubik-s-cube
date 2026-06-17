import kociemba


def solve(cube_string):
    solution = kociemba.solve(cube_string)
    return solution.strip().split()


def print_solution(moves):
    print(" ".join(moves))
