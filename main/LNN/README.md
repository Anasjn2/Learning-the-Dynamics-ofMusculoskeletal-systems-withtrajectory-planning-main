# Thesis Work: Trajectory Generation for Robotic Arms Using System Identification with Neural Networks

This repository contains code for generating trajectories for robotic arms using system identification with neural networks. The theory and implementation are based on the work done in [Miles Cranmer's lagrangian_nns repository](https://github.com/MilesCranmer/lagrangian_nns).

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- NumPy
- SciPy
- Matplotlib
- JAX



### Usage

1. Generate trajectories using `generatemuscletraj.py` for the arm with muscles or `generatetrajnomuscle.py` for the arm without muscles, and specify the chosen hyperparameters.

2. Start the system identification task with `JaxLearning.py`.

3. Import learned neural network `params.npy` using `TrajectoryPlanning.py`, specify desired goal positions, and obtain trajectory.

## Acknowledgments

- [Miles Cranmer's lagrangian_nns repository](https://github.com/MilesCranmer/lagrangian_nns) for providing the theory and implementation for this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.