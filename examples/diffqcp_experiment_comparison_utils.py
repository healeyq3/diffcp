from dataclasses import dataclass, field
from typing import Optional, Callable, Union
import time
import os
import json

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import spmatrix, sparray
import cvxpy as cvx
import matplotlib.pyplot as plt
from diffcp import solve_and_derivative

def _convert(obj):
    """Helper function for saving cone dictionaries"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@dataclass
class DiffcpData:

    A : Union[spmatrix, sparray]
    c: np.ndarray
    b: np.ndarray
    cone_dict: dict[str, int | list[int]]


@dataclass
class GradDescTestResult:

    passed : bool
    num_iterations : int
    obj_traj : np.ndarray
    for_qcp: bool
    learning_time: float
    lsqr_residuals: np.ndarray | None = None
    improvement_factor: float = field(init=False)

    def __post_init__(self):
        self.improvement_factor = self.obj_traj[0] / self.obj_traj[-1]

    def plot_obj_traj(self, savepath: str) -> None:

        if self.obj_traj is None:
            raise ValueError("obj_traj is None. Cannot plot.")

        plt.figure(figsize=(8, 6))
        plt.plot(range(self.num_iterations), self.obj_traj, label="Objective Trajectory")
        # plt.plot(range(self.num_iterations), np.log(self.obj_traj), label="Objective Trajectory")
        # if self.lsqr_residuals is not None:
            # plt.plot(range(self.num_iterations), np.log(self.lsqr_residuals), label="LSQR residuals")
        plt.xlabel("num. iterations")
        # plt.ylabel("$f_0(p^{k}) = 0.5 \\| z(p) - z^{\\star} \\|^2$")
        plt.ylabel("Objective function")
        plt.legend()
        if self.for_qcp:
            plt.title(label="diffqcp")
        else:
            plt.title(label="diffcp")
        plt.savefig(savepath)
        plt.close()

    def save_result(self, savepath: str, experiment_name: str, experiment_count: int=0, verbose: bool=False) -> None:
        log_path = os.path.join(savepath, f"logs/experiment_{experiment_count}_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_content = []
        log_content.append(
            f"  diffcp learning experiment:\n"
            f"  Iterations: {self.num_iterations}\n"
            f"  Learning time: {self.learning_time}\n"
            f"  Improvement factor: {self.improvement_factor}\n"
            f"  Final loss: {self.obj_traj[-1]}\n"
        )

        if verbose:
            print(log_content)
            print("=== ===")
        
        with open(log_path, "a") as f:
            f.write("\n".join(log_content))
            f.write("\n")
            f.write("=== ===\n")

        if self.lsqr_residuals is not None:
            lsqr_dir = os.path.join(savepath, f"data/run_{experiment_count}/qcp")
            os.makedirs(lsqr_dir, exist_ok=True)
            np.save(os.path.join(lsqr_dir, "lsqr_residuals.npy"), self.lsqr_residuals)


@dataclass
class GradDescTestHelper:

    # problem_generator: Callable[[], cvx.Problem]
    problem_data_path: str
    # passing criteria? (multiple?)
    verbose: bool = False
    save_dir: Optional[str] = None
    
    # post init attributes
    linear_cone_dict: dict[str, int | list[int]] = field(init=False)
    diffcp_cp: DiffcpData = field(init=False)
    target_x_cp: np.ndarray = field(init=False)
    target_y_cp: np.ndarray = field(init=False)
    target_s_cp: np.ndarray = field(init=False)
    reset_counter: int = field(default=0, init=False)

    def __post_init__(self):

        self._reset_problems()
    
    def _reset_problems(self):
        self.reset_counter += 1
        self.cp_has_descended: bool = False

        # grab target problem data for CP.
        # cp_data = data_from_cvxpy_problem_linear(target_problem)
        # target_x_cp, target_y_cp, target_s_cp, _, _ = solve_and_derivative(cp_data[0], cp_data[2], cp_data[1], cp_data[3], solve_method='CLARABEL')
        # use save path
        data_path = os.path.join(self.problem_data_path, f"run_{self.reset_counter}/cp")
        self.target_x_cp = np.load(os.path.join(data_path, "x_target.npy"))
        self.target_y_cp = np.load(os.path.join(data_path, "y_target.npy"))
        self.target_s_cp = np.load(os.path.join(data_path, "s_target.npy"))
        
        # Now grab starting data for the learning problem
        # cp_data = data_from_cvxpy_problem_linear(initial_problem)
        # use save path
        A = sparse.load_npz(os.path.join(data_path, "A_initial.npz"))
        assert isinstance(A, sparse.csc_matrix)
        b = np.load(os.path.join(data_path, "b_initial.npy"))
        c = np.load(os.path.join(data_path, "c_initial.npy"))
        with open(os.path.join(data_path, "scs_cones.json")) as f:
            scs_cones = json.load(f)
        self.diffcp_cp = DiffcpData(A=A, c=c, b=b, cone_dict=scs_cones)


    def cp_grad_desc(
        self,
        num_iter: int = 150,
        step_size: float = 0.15,
        improvement_factor: float = 1e-1, #10x improvement
        fixed_tol: float = 1e-3
    ) -> GradDescTestResult:
        
        if self.cp_has_descended:
            self._reset_problems()
            self.cp_has_descended = True

        curr_iter = 0
        optimal = False
        f0s = np.zeros(num_iter)
        lsqr_residuals = np.zeros(num_iter)
        
        def f0(x, y, s) -> float:
            return (0.5 * np.linalg.norm(x - self.target_x_cp)**2 + 0.5 * np.linalg.norm(y - self.target_y_cp)**2
                    + 0.5 * np.linalg.norm(s - self.target_s_cp)**2)

        start_time = time.perf_counter()
        
        while curr_iter < num_iter:
            
            A = self.diffcp_cp.A
            c = self.diffcp_cp.c
            b = self.diffcp_cp.b

            xk, yk, sk, _, DT = solve_and_derivative(A, b, c, self.diffcp_cp.cone_dict, solve_method='CLARABEL')

            f0k = f0(xk, yk, sk)

            f0s[curr_iter] = f0k
            curr_iter += 1

            if curr_iter > 1 and ((f0k / f0s[0]) < improvement_factor or f0k < fixed_tol):
                optimal = True
                break

            dA, db, dc, resid = DT(xk - self.target_x_cp, yk - self.target_y_cp, sk - self.target_s_cp, return_resid=True)
            lsqr_residuals[curr_iter - 1] = resid

            self.diffcp_cp.A += -step_size * dA
            self.diffcp_cp.c += -step_size * dc
            self.diffcp_cp.b += -step_size * db

        end_time = time.perf_counter()
        
        f0_traj = f0s[0:curr_iter]
        residuals = lsqr_residuals[0:num_iter]
        del f0s
        del lsqr_residuals
        return GradDescTestResult(
                passed=optimal, num_iterations=curr_iter, obj_traj=f0_traj, for_qcp=False, learning_time=(end_time - start_time), lsqr_residuals=residuals
            )