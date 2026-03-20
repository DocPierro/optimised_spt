# Exact Language Computation for Stochastic Process Trees via Discrete-Time Markov Chain Construction

This repository hosts the implementation of the methodology outlined in the paper titled "Exact Language Computation for Stochastic Process Trees via Discrete-Time Markov Chain Construction". The proposed code enables the construction of a LADTMC from a stochastic process tree and exploits this representation to perform an efficient optimization of probability parameters for stochastic process tree mining in the context of stochastic process discovery.

## Contents
**Scripts:** Contains the necessary scripts to execute an optimized search of probability parameters for stochastic process tree mining.<br/>
&emsp; ***language/Eventlog*** contains the necessary to import and manipulate an eventlog.<br/>
&emsp; ***model/ProcessTree*** provides the core structures and operations for handling both stochastic and non-stochastic process trees. In particular, this module implements the semantic rules described in the paper, including the construction of the corresponding DTMC representation (`to_dtmc()`).<br/>
&emsp; ***model/MarkovChain*** provides the core data structures and algorithms for representing and manipulating Markov chain models, in particular Discrete-Time Markov Chains (DTMCs)<br/>
&emsp; ***cal/Optimizer_PT*** contains the implementation of the optimized probability search.<br/>

**Data:** Includes a selection of event logs in ***.xes*** utilized for experimentation.<br/>
&emsp; ***rl_data/BPIC13_closed.xes*** -> [https://data.4tu.nl/datasets/1987a2a6-9f5b-4b14-8d26-ab7056b17929](https://data.4tu.nl/datasets/1987a2a6-9f5b-4b14-8d26-ab7056b17929)<br/>
&emsp; ***rl_data/BPIC13_incidents.xes*** -> [https://data.4tu.nl/datasets/1987a2a6-9f5b-4b14-8d26-ab7056b17929](https://data.4tu.nl/datasets/0fc5c579-e544-4fab-9143-fab1f5192432)<br/>
&emsp; ***rl_data/BPIC13_open.xes*** -> [https://data.4tu.nl/datasets/7aafbf5b-97ae-48ba-bd0a-4d973a68cd35](https://data.4tu.nl/datasets/7aafbf5b-97ae-48ba-bd0a-4d973a68cd35)<br/>
&emsp; ***rl_data/BPIC17_offerlog.xes*** -> [https://data.4tu.nl/datasets/cc497753-1175-41f6-a107-425787c54266](https://data.4tu.nl/datasets/cc497753-1175-41f6-a107-425787c54266)<br/>
&emsp; ***rl_data/BPIC20_dd.xes*** -> [https://data.4tu.nl/datasets/6a0a26d2-82d0-4018-b1cd-89afb0e8627f](https://data.4tu.nl/datasets/6a0a26d2-82d0-4018-b1cd-89afb0e8627f)<br/>
&emsp; ***rl_data/BPIC20_rfp.xes*** -> [https://data.4tu.nl/datasets/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc](https://data.4tu.nl/datasets/a6f651a7-5ce0-4bc6-8be1-a7747effa1cc)<br/>
&emsp; ***rl_data/BPIC12.xes*** -> [https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204)<br/>
Data are stored in the rl_data.zip file and must be unzipped for use.

**Results:** Presents the outcome of a series of tests as outlined in the associated paper.<br/>
&emsp; ***result***: Contains the series of results obtained from running the optimizer on logs, including computation time and metrics values.<br/>

## Installation
This project requires Python 3.9 or higher. The required libraries and their versions are listed in the requirements file. Follow these steps to set up the codebase:

1. Clone this repository to your local machine:<br/>
2. Navigate to the project directory:<br/>
3. Set up a virtual environment using Conda (optional but recommended):<br/>
&emsp; a. Create a new Conda environment:<br/>
&emsp; &emsp; `conda create -n myenv python=3.9`<br/>
&emsp; b. Activate the Conda environment:<br/>
&emsp; &emsp; `conda activate myenv`<br/>
4. Install the required libraries from the ***requirements.txt*** file:<br/>
&emsp; `pip install -r "requirements.txt"`<br/>
5. Modify the ***main.py*** script according to your needs.
6. Launch the script:<br/>
&emsp; `python main.py`

## How to use
Using the ***main.py*** script:
1. Import an event log from an ***.xes*** file:<br/>
&emsp; `ev = Eventlog(filename)`<br/>
2. Construct a process tree using discovery algorithms:<br/>
&emsp; `pt = ev.discover_pt_inductive()`<br/>
3. Specify the optimization parameters:<br/>
&emsp; ***m***: Specify the metric to minimize with (***KLD*** or ***rEMD***).<br/>
&emsp; ***solver***: Specify the minimization method (***L-BFGS-B*** or ***TNC*** or ***Powell*** or ***Nelder-Mead***).<br/>
&emsp; ***starting_vector***: Specifies the initial probability vector used for the optimisation procedure. If set to `None`, the optimisation starts from a randomly generated vector.<br/>
&emsp; ***nw0***: Specify the number of randomly chosen probability vectors considered as a starting point for minimization.<br/>
4. Launch an optimization-based estimator:<br/>
&emsp; &emsp; `opt_pt = Optimizer_PT(ev, pt)`<br/>
&emsp; &emsp; `opt_pt.estimate_exact(m, solver, starting_vector, nw0)`<br/>

## Contact

For any inquiries or assistance, please contact #####
