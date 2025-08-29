# `README.md`

# Bias-Adjusted LLM Agents for Human-Like Decision-Making

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Type Checking: mypy](https://img.shields.io/badge/type_checking-mypy-blue)](http://mypy-lang.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%23025596?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?style=flat&logo=openai&logoColor=white)](https://openai.com/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.18600-b31b1b.svg)](https://arxiv.org/abs/2508.18600v1)
[![Research](https://img.shields.io/badge/Research-Computational%20Social%20Science-green)](https://github.com/chirindaopensource/bias_adjusted_LLM_agents_human_like_decision_making)
[![Discipline](https://img.shields.io/badge/Discipline-Behavioral%20Economics%20%26%20NLP-blue)](https://github.com/chirindaopensource/bias_adjusted_LLM_agents_human_like_decision_making)
[![Methodology](https://img.shields.io/badge/Methodology-Agent--Based%20Modeling-orange)](https://github.com/chirindaopensource/bias_adjusted_LLM_agents_human_like_decision_making)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/bias_adjusted_LLM_agents_human_like_decision_making)

**Repository:** `https://github.com/chirindaopensource/bias_adjusted_LLM_agents_human_like_decision_making`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Bias-Adjusted LLM Agents for Human-Like Decision-Making via Behavioral Economics"** by:

*   Ayato Kitadai
*   Yusuke Fukasawa
*   Nariaki Nishino

The project provides a complete, end-to-end computational framework for simulating heterogeneous populations of human-like agents in economic games. It implements the paper's novel persona-based approach to condition Large Language Models (LLMs) with empirical, individual-level behavioral data, thereby adjusting their intrinsic biases to better align with observed human decision-making patterns. The goal is to provide a transparent, robust, and extensible toolkit for researchers to replicate, validate, and build upon the paper's findings in the domain of computational social science and agent-based modeling.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: run_entire_project](#key-callable-run_entire_project)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "Bias-Adjusted LLM Agents for Human-Like Decision-Making via Behavioral Economics." The core of this repository is the iPython Notebook `bias_adjusted_LLM_agents_human_like_decision_making_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation to the final execution of a full suite of robustness checks.

The use of LLMs to simulate human behavior is a promising but challenging frontier. Off-the-shelf models often exhibit behavior that is too rational or systematically biased in ways that do not reflect the diversity of a real human population. This project implements the paper's innovative solution: injecting individual-level behavioral traits from the Econographics dataset into LLMs to create a heterogeneous population of "digital twins."

This codebase enables users to:
-   Rigorously validate and cleanse the input persona and benchmark experimental data.
-   Set up a robust, multi-provider infrastructure for interacting with leading LLMs (OpenAI, Google, Anthropic).
-   Execute the full 3x3x2 factorial simulation of the Ultimatum Game with 1,000 agents per condition.
-   Leverage a fault-tolerant simulation engine with checkpointing and resume capabilities.
-   Aggregate individual agent decisions into population-level probability distributions.
-   Quantitatively validate simulation results against empirical data using the Wasserstein distance.
-   Generate high-fidelity replications of the paper's key figures and tables.
-   Conduct a comprehensive suite of robustness checks to test the stability of the findings.

## Theoretical Background

The implemented methods are grounded in behavioral economics, game theory, and natural language processing.

**1. The Ultimatum Game:**
The experimental testbed is the classic Ultimatum Game. A Proposer offers a split of 100 coins, and a Responder can either accept or reject. The payoff `(u_P, u_R)` is:

$$
(u_P, u_R) = 
\begin{cases} 
(100 - s, s) & \text{if Responder accepts} \\
(0, 0) & \text{if Responder rejects} 
\end{cases}
$$

where `s` is the amount offered to the Responder. While classical game theory predicts a minimal offer, human behavior systematically deviates, showing preferences for fairness. This well-documented gap makes it an ideal benchmark for testing the human-likeness of LLM agents.

**2. Persona-Based Conditioning:**
The core innovation is to move beyond generic LLM agents by providing them with a "persona" in their context window. This is achieved by constructing a text block from the **Econographics dataset**, which contains 21 measured behavioral indicators (e.g., Risk Aversion, Reciprocity, Patience) for 1,000 individuals. By assigning each LLM agent a unique persona from this dataset, the simulation aims to create a population whose distribution of behavioral biases mirrors the human sample.

**3. Wasserstein Distance for Validation:**
The alignment between the simulated distribution of offers (`P_sim`) and the empirical human distribution (`P_human`) is quantified using the Wasserstein-1 distance. For 1D discrete distributions, this is the sum of the absolute differences between the cumulative distribution functions (CDFs):

$$
W_1(P_{\text{sim}}, P_{\text{human}}) = \sum_{k=0}^{100} |\text{CDF}_{\text{sim}}(k) - \text{CDF}_{\text{human}}(k)|
$$

A lower distance indicates a better alignment between the simulated and real-world behavior.

## Features

The provided iPython Notebook (`bias_adjusted_LLM_agents_human_like_decision_making_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Task-Based Architecture:** The entire pipeline is broken down into 8 distinct, modular tasks, from data validation to robustness analysis.
-   **Professional-Grade Data Validation:** A comprehensive validation suite ensures all inputs (data and configurations) conform to the required schema before execution.
-   **Auditable Data Cleansing:** A non-destructive cleansing process that handles missing values and outliers, returning a detailed log of all transformations.
-   **Multi-Provider LLM Infrastructure:** A robust, fault-tolerant system for interacting with OpenAI, Google (Gemini), and Anthropic (Claude) APIs, featuring exponential backoff retries and fallback parsing.
-   **Resumable Simulation Engine:** A checkpointing system allows the 18,000-decision simulation to be stopped and resumed without loss of progress.
-   **Rigorous Statistical Analysis:** Implements correct aggregation of individual decisions into statistical distributions and applies the Wasserstein distance and alternative metrics (KL, JS Divergence).
-   **High-Fidelity Visualization:** Generates publication-quality replications of the paper's main figures.
-   **Comprehensive Robustness Suite:** A meta-orchestrator to systematically test the sensitivity of the results to changes in sample size, persona definitions, and prompt wording.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Input Data Validation (Task 1):** Ingests and rigorously validates all raw data and configuration files.
2.  **Data Preprocessing (Task 2):** Cleanses and aligns the persona and benchmark datasets.
3.  **LLM Infrastructure Setup (Task 3):** Authenticates and initializes clients for all LLM providers.
4.  **Simulation Execution (Task 4):** Runs the full 3x3x2 factorial simulation of the Ultimatum Game.
5.  **Data Aggregation (Task 5):** Transforms raw decisions into population-level statistical distributions.
6.  **Quantitative Validation (Task 6):** Computes Wasserstein distances to measure alignment with human data.
7.  **Results Visualization (Task 7):** Generates high-fidelity replications of the paper's figures and tables.
8.  **Robustness Analysis (Task 8):** Provides a master function to run the full suite of robustness checks.

## Core Components (Notebook Structure)

The `bias_adjusted_LLM_agents_human_like_decision_making_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 8 major tasks.

## Key Callable: run_entire_project

The central function in this project is `run_entire_project`. It orchestrates the entire analytical workflow, providing a single entry point for running the baseline study replication and the advanced robustness checks.

```python
def run_entire_project(
    personas_data_path: str,
    benchmark_data_path: str,
    study_configurations: Dict[str, Any],
    output_dir: str,
    force_rerun_simulation: bool = False,
    run_robustness_checks: bool = True,
    # ... other robustness configurations
) -> Dict[str, Any]:
    """
    Executes the entire research project, including the main study pipeline
    and all subsequent robustness analyses.
    """
    # ... (implementation is in the notebook)
```

## Prerequisites

-   Python 3.9+
-   API keys for OpenAI, Google Cloud Platform, and access to Anthropic models on Vertex AI.
-   Core dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `tqdm`, `openai`, `google-cloud-aiplatform`, `anthropic`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/bias_adjusted_LLM_agents_human_like_decision_making.git
    cd bias_adjusted_LLM_agents_human_like_decision_making
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy matplotlib tqdm openai "google-cloud-aiplatform>=1.38.1" "anthropic[vertex]" ipython
    ```

4.  **Set up API Credentials:**
    -   Export your API keys as environment variables.
        ```sh
        export OPENAI_API_KEY="sk-..."
        export GCP_PROJECT="your-gcp-project-id"
        ```
    -   Authenticate with Google Cloud for Vertex AI access.
        ```sh
        gcloud auth application-default login
        ```

## Input Data Structure

The pipeline requires two CSV files with specific structures, which are rigorously validated by the first task.
1.  **Personas Data (`personas.csv`):** A CSV with 1000 rows and 26 columns: `participant_id`, 21 specific behavioral indicators (e.g., `Reciprocity:High`), `age`, `gender`, `country_of_residence`, and `crt_score`.
2.  **Benchmark Data (`benchmark.csv`):** A CSV with 1000 rows and 3 columns: `interaction_id`, `proposer_offer`, and `responder_decision`.

A mock data generation script is provided in the main notebook to create valid example files for testing the pipeline.

## Usage

The `bias_adjusted_LLM_agents_human_like_decision_making_draft.ipynb` notebook provides a complete, step-by-step guide. The core workflow is:

1.  **Prepare Inputs:** Create your data CSVs or use the provided mock data generator. Ensure your API keys are set as environment variables.
2.  **Execute Pipeline:** Call the grand master orchestrator function.

    ```python
    # This single call runs the baseline analysis and all configured robustness checks.
    final_project_artifacts = run_entire_project(
        personas_data_path="./data/personas.csv",
        benchmark_data_path="./data/benchmark.csv",
        study_configurations=study_configurations,
        output_dir="./project_outputs",
        force_rerun_simulation=False,
        run_robustness_checks=True,
        sample_sizes_to_test=[500, 750]
    )
    ```
3.  **Inspect Outputs:** Programmatically access any result from the returned dictionary. For example, to view the main results table:
    ```python
    main_results_table = final_project_artifacts['main_study_results']['results_table_data']
    print(main_results_table)
    ```

## Output Structure

The `run_entire_project` function returns a single, comprehensive dictionary with two top-level keys:
-   `main_study_results`: A dictionary containing all artifacts from the primary study replication (cleansed data, logs, raw simulation results, aggregated distributions, final tables, etc.).
-   `robustness_analysis_results`: A dictionary containing the summary tables from each of the executed robustness checks.

All generated files (checkpoints, plots) are saved to the specified `output_dir`.

## Project Structure

```
bias_adjusted_LLM_agents_human_like_decision_making/
│
├── bias_adjusted_LLM_agents_human_like_decision_making_draft.ipynb  # Main implementation notebook
├── requirements.txt                                                 # Python package dependencies
├── LICENSE                                                          # MIT license file
└── README.md                                                        # This documentation file
```

## Customization

The pipeline is highly customizable via the `study_configurations` dictionary and the arguments to the `run_entire_project` function. Users can easily modify all relevant parameters, such as the list of models to test, the definitions of persona configurations, prompt templates, and the specific robustness checks to perform.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:

-   **Automated Report Generation:** Creating a function that takes the final `project_artifacts` dictionary and generates a full PDF or HTML report summarizing the findings, including tables, figures, and interpretive text.
-   **Generalization to Other Games:** Refactoring the game-specific logic (e.g., prompt templates, decision validation) to allow the pipeline to be easily adapted to other classic economic games like the Prisoner's Dilemma or Public Goods Game.
-   **Advanced Persona Engineering:** Implementing more sophisticated methods for selecting or weighting behavioral traits, potentially using feature selection algorithms to identify the most impactful traits for aligning agent behavior.
-   **Formal Scientific Integrity Report:** Implementing the functions to perform and document a systematic bias analysis and generate a comprehensive limitations section, as outlined in the original Task 8.3.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{kitadai2025bias,
  title={Bias-Adjusted LLM Agents for Human-Like Decision-Making via Behavioral Economics},
  author={Kitadai, Ayato and Fukasawa, Yusuke and Nishino, Nariaki},
  journal={arXiv preprint arXiv:2508.18600},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Python Implementation of "Bias-Adjusted LLM Agents for Human-Like Decision-Making via Behavioral Economics". 
GitHub repository: https://github.com/chirindaopensource/bias_adjusted_LLM_agents_human_like_decision_making
```

## Acknowledgments

-   Credit to Ayato Kitadai, Yusuke Fukasawa, and Nariaki Nishino for their innovative and clearly articulated research.
-   Thanks to the developers of the scientific Python ecosystem (`numpy`, `pandas`, `scipy`, `matplotlib`, etc.) and the teams at OpenAI, Google, and Anthropic for their powerful open-source tools and proprietary models.

--

*This README was generated based on the structure and content of `bias_adjusted_LLM_agents_human_like_decision_making_draft.ipynb` and follows best practices for research software documentation.*
