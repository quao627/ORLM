# gurobipy
from gurobipy import GRB
import gurobipy as gp
import os
from contextlib import redirect_stdout, redirect_stderr


def execute_gurobi(code_str):
    """Execute Gurobi code and return optimal value or error type."""
    try:
        exec_globals = {'gp': gp, 'GRB': GRB}
        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            exec(code_str, exec_globals)
        return {"success": True, "value": exec_globals.get('model', gp.Model()).objVal}
    except Exception as e:
        return {"success": False, "value": str(e)}
    

def test_equivalence(solution1, solution2):
    if abs(solution1 - solution2) < 1e-3:
        return True
    else:
        return False
    
    
def test_optimality(code, ground_truth):
    result = execute_gurobi(code)
    if not result["success"]:
        return "runtime_error"
    if test_equivalence(result["value"], ground_truth):
        return "correct"
    else:
        return "wrong_answer"
