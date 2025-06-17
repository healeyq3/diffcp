import os
from datetime import datetime

import numpy as np

from examples.diffqcp_experiment_comparison_utils import GradDescTestHelper

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(os.path.dirname(__file__), f"results/results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

if __name__ == "__main__":

    # np.random.seed(13)
    
    num_iter = 500
    step_size = 1e-5
    num_loop_trials = 10

    # # smaller parameters for testing
    # # m = 20
    # # n = 10
    # # num_iter = 20
    # # step_size = 1e-5
    # # num_loop_trials = 3

    # # TODO(quill): add try catch blocks?

    # # === ===

    # data_load_path = "/home/quill/diffqcp/experiments/results/results_20250615-092317/group_lasso/data"
    # exp_results_dir = os.path.join(results_dir, "group_lasso")
    # os.makedirs(exp_results_dir, exist_ok=True)
    # exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    # os.makedirs(exp_results_dir_plots, exist_ok=True)
    # experiment = GradDescTestHelper(data_load_path, save_dir=exp_results_dir)
    # for i in range(1, num_loop_trials+1):
    #     print(f"=== starting diffcp group lasso experiment {i} / {num_loop_trials} ===")
    #     experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     # save results
    #     experiment_diffcp_result.save_result(exp_results_dir, experiment_name="group lasso", experiment_count=i, verbose=True)
    #     diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
    #     experiment_diffcp_result.plot_obj_traj(diffcp_save_path)

    # # === ===

    # data_load_path = "/home/quill/diffqcp/experiments/results/results_20250615-092317/portfolio/data"
    # exp_results_dir = os.path.join(results_dir, "portfolio")
    # os.makedirs(exp_results_dir, exist_ok=True)
    # exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    # os.makedirs(exp_results_dir_plots, exist_ok=True)
    # experiment = GradDescTestHelper(data_load_path, save_dir=exp_results_dir)
    # for i in range(1, num_loop_trials+1):
    #     print(f"=== starting diffcp portfolio experiment {i} / {num_loop_trials} ===")
    #     experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     # save results
    #     experiment_diffcp_result.save_result(exp_results_dir, experiment_name="portfolio", experiment_count=i, verbose=True)
    #     diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
    #     experiment_diffcp_result.plot_obj_traj(diffcp_save_path)

    # # === ===
    
    # n = 20
    # p = 10

    # data_load_path = "/home/quill/diffqcp/experiments/results/results_20250615-092317/sdp/data"
    # exp_results_dir = os.path.join(results_dir, "sdp")
    # os.makedirs(exp_results_dir, exist_ok=True)
    # exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    # os.makedirs(exp_results_dir_plots, exist_ok=True)
    # experiment = GradDescTestHelper(data_load_path, save_dir=exp_results_dir)
    # for i in range(1, num_loop_trials+1):
    #     print(f"=== starting diffcp sdp experiment {i} / {num_loop_trials} ===")
    #     experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     # save results
    #     experiment_diffcp_result.save_result(exp_results_dir, experiment_name="sdp", experiment_count=i, verbose=True)
    #     diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
    #     experiment_diffcp_result.plot_obj_traj(diffcp_save_path)

    # === ===

    n=20

    data_load_path = "/home/quill/diffqcp/experiments/results/results_20250615-092317/robust_mvdr/data"
    exp_results_dir = os.path.join(results_dir, "robust_mvdr")
    os.makedirs(exp_results_dir, exist_ok=True)
    exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    os.makedirs(exp_results_dir_plots, exist_ok=True)
    experiment = GradDescTestHelper(data_load_path, save_dir=exp_results_dir)
    for i in range(1, num_loop_trials+1):
        print(f"=== starting diffcp mvdr experiment {i} / {num_loop_trials} ===")
        experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
        # save results
        experiment_diffcp_result.save_result(exp_results_dir, experiment_name="mvdr", experiment_count=i, verbose=True)
        diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
        experiment_diffcp_result.plot_obj_traj(diffcp_save_path)