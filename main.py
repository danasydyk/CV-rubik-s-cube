import argparse
from scanner import run_scanner
from solver import solve, print_solution


def main(debug=False):
    result = run_scanner(debug=debug)
    if result is None:
        return

    cube_string, _ = result

    try:
        moves = solve(cube_string)
    except Exception as e:
        print(f"Solver error: {e}")
        return

    print_solution(moves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(debug=args.debug)
