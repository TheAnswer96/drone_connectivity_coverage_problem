# Python Code for BVLoS Drones Path Optimization

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository contains the Python implementation of the algorithms and experiments described in our paper, **"Optimizing Connectivity and Coverage for UAV Paths in BVLoS Operations in Ad Hoc Networks"**, which is currently under review.

Unmanned Aerial Vehicles (UAVs), or drones, have transformative potential in civilian applications.
However, their operational range is often limited by strict aviation regulations requiring Visual Line of Sight (\vlos). 
Our work addresses this limitation by enabling Beyond Visual Line of Sight (BVLoS) operations, ensuring continuous remote monitoring through existing ad hoc wireless ground infrastructure.

This repository provides:
- Implementations of polynomial-time algorithms for solving **single-trajectory problems**.
- Solutions for the NP-hard **multi-trajectory problem** using heuristic approaches.
- Experimentation scripts to evaluate algorithmic performance on various network configurations, including random geometric graphs and regular grids.

---

## Features

- **Single-Trajectory Optimization**:
  - Minimize antenna hops (eccentricity) for BVLoS coverage.
  - Balance between minimizing coverage and reducing eccentricity.

- **Multi-Trajectory Optimization**:
  - Heuristic approaches for NP-hard multi-path connectivity problems.

- **Experimental Evaluation**:
  - Generate random network configurations (geometric graphs, grids).
  - Evaluate algorithmic performance across diverse setups.

---

## Installation

Clone the repository and install the required Python dependencies:
```bash
git clone https://github.com/TheAnswer96/drone_connectivity_coverage_problem.git
cd drone_connectivity_coverage_problem
pip install matplotlib networkx pandas
Python main.py
```
---
## Citation

If you use this repository, please cite our paper when it becomes available. For now, you can use the following placeholder:

```bibtex
@article{Betti2024BVLoS,
  title={Optimizing Connectivity and Coverage for UAV Paths in BVLoS Operations in Ad Hoc Networks},
  author={Betti Sorbelli, Francesco and Ghobadi, Sajjad and Palazzetti, Lorenzo and Pinotti, Cristina M},
  journal={Under Review},
  year={2024},
  note={Available soon}
}
```

## Contact Us

For questions, feedback, or collaboration opportunities, feel free to reach out to us:

- **Francesco Betti Sorbelli**: [francesco.bettisorbelli@unipg.it](mailto:francesco.bettisorbelli@unipg.it)
- **Sajjad Ghobadi**: [sajjad.ghobadibabi@unipg.it](mailto:sajjad.ghobadibabi@unipg.it)
- **Lorenzo Palazzetti**: [lorenzo.palazzetti@unipg.it](mailto:lorenzo.palazzetti@unipg.it)
- **Cristina M. Pinotti**: [cristina.pinotti@unipg.it](mailto:cristina.pinotti@unipg.it)

