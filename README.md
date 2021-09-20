# Models and algorithms for solving variants of the continuous resource scheduling problem

Implementation and comparison of a number of different approaches for solving a resource assignment scheduling problem. 

## Table of Contents

- [Models and algorithms for solving variants of the continuous resource scheduling problem](#models-and-algorithms-for-solving-variants-of-the-continuous-resource-scheduling-problem)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
    - [License](#license)
    - [Attribution and academic use](#attribution-and-academic-use)
    - [Contact](#contact)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Generating instances](#generating-instances)
    - [Solving a model for an instance](#solving-a-model-for-an-instance)
  - [Links](#links)
  - [Notes](#notes)

## About the Project

**Date**: April 2021 - ...

**Contributors**:

- [R.J.J. Brouwer](https://www.uu.nl/staff/RJJBrouwer) ([@RoelBrouwer](https://github.com/RoelBrouwer))

### License

The code in this project is released under _[License name](LICENSE)._
<!-- Update later -->

### Attribution and academic use

_Placeholder._
<!-- Update later -->

### Contact

Any questions or remarks can be submitted in the [issue tracker](https://github.com/RoelBrouwer/continuousresource/issues) or be directed to R.J.J. Brouwer (for contact details, see above).

## Getting Started

### Prerequisites
To use the scripts in this repository with [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) or [Gurobi](https://www.gurobi.com/), you need to have a (licensed) copy of the appropriate software installed on your computer. Instructions are listed below for each of the solvers.

In the current version, solvers other than CPLEX are only supported for a subset of all functionality.

#### CPLEX
1. Obtain a copy of [IBM's ILOG CPLEX Optimization Studio](https://www.ibm.com/products/ilog-cplex-optimization-studio). At the time of writing, v12.10.0 is the most recent version.
2. The installation wizard will tell you what to do to expose the CPLEX Python API. If not, follow [the CPLEX Python setup guide](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

#### Gurobi
1. Obtain a copy of [Gurobi Optimizer](https://www.gurobi.com/downloads/). At the time of writing, v9.0.2 is the most recent version.
2. After installation, the Python API of Gurobi should be ready for use.
    - If you start a "fresh" [conda](https://docs.conda.io/) environment, be aware that the current version of `gurobi` requires your Python version to be `<3.8`.
    - You can get around this by manually setting up the `gurobi` package from the source of you Gurobi Optimizer copy. See [this community question on using Gurobi with higher Python versions](https://support.gurobi.com/hc/en-us/community/posts/360059881591-Gurobi-with-python-version-3-8) and its answers (in particular [this one](https://support.gurobi.com/hc/en-us/community/posts/360059881591/comments/360012744731)) for more details.

### Installation

It is recommended that you use [conda](https://docs.conda.io/) to manage your environment. If you choose to configure your environment manually, be sure to install the dependencies listed in the appropriate `.yml` files, and ignore steps 1 and 4.

1. Install conda by following the [installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Clone this git repository into a directory on your local system.

with HTTPS:
```sh
git clone https://github.com/RoelBrouwer/continuousresource.git <RELATIVE_FOLDER_PATH>
```

or SSH :
```sh
git clone git@github.com:RoelBrouwer/continuousresource.git <RELATIVE_FOLDER_PATH>
```

3. Open the Anaconda prompt (Windows) or a terminal window (MacOS, Linux) and navigate to the installation folder.
4. Create a new environment based on the appropriate `.yml`-file(s) ([`cplexenv`](cplexenv.yml), [`instancegeneration`](instancegeneration.yml) or [`solve`](solve.yml))
```sh
conda env create --file <ENVNAME>.yml
```
5. Install the `continuousresource` package by navigating into the directory where the repository is located and running:
```sh
python setup.py install
```

If you have set up the solvers as instructed in the [prerequisites section](#prerequisites), you are all set to use the scripts in this repository.

## Usage

You can use the package by importing any of the functions or classes from the continuousresource package. Look at the [Placeholder for a link to generated documentation]() for detailed technical documentation.

Example uses and related scripts are located in the [`continuousresource/scripts` folder](continuousresource/scripts):
- [`instancegeneration.py`](continuousresource/scripts/instancegeneration.py): generates a single problem instance
- [`lp_stresstest.py`](continuousresource/scripts/lp_stresstest.py): creates and solves an LP based on a number of randomly generated instances
- [`multipleinstancegeneration.py`](continuousresource/scripts/multipleinstancegeneration.py): like `instancegeneration.py`, but generating instances in bulk for a large number of parameter combinations
- [`runtests.py`](continuousresource/scripts/runtests.py): solves the same problem instance with a number of different techniques and compares the results.
- [`solveinstance.py`](continuousresource/scripts/instancegeneration.py): solve a single problem instance using a single approach
- [`test_simulated_annealing.py`](continuousresource/scripts/test_simulated_annealing.py): testrun implementing a simulated annealing approach


Two quick example uses are described in detail below.

### Generating instances

1. Open the Anaconda prompt (Windows) or a terminal window (MacOS, Linux) and navigate to the installation folder.
2. Activate the [`instancegeneration`](instancegeneration.yml) environment
```sh
conda activate instancegeneration
```
3. Run the instancegeneration script with the appropriate parameters, e.g.:
```sh
python scripts/instancegeneration.py -n 5 -m 1 -k 2 -f both -p data/smallinstance
```
Note: make sure to create the `data` directory first, when running this example command as-is.

### Solving a model for an instance

1. Open the Anaconda prompt (Windows) or a terminal window (MacOS, Linux) and navigate to the installation folder.
2. Activate the [`solve`](solve.yml) environment
```sh
conda activate solve
```
3. Run the solveinstance script with the appropriate parameters, e.g.:
```sh
python scripts/solveinstance.py -f csv -p data/smallinstance -m dof -e 1 -s glpk -o results
```
Note: make sure to create the `results` directory first, when running this example command as-is.

## Links

_Placeholder._

## Notes

_Placeholder._