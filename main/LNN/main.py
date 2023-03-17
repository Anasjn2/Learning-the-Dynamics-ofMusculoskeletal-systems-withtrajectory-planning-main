# -*- coding: utf-8 -*-


# Main program
import subprocess

# Generate trajectories
subprocess.run(["python", "generatetrajnomuscle.py"], check=True)
subprocess.run(["python", "JaxLearning.py"], check=True)
subprocess.run(["python", "TrajectoryPlanning.py"], check=True)