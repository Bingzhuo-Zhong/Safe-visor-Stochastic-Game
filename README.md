
## Synthesizing Safe-visor Architecture for Stochastic Games

Here is a repository of the codes for synthesizing abstraction-based Safe-visor Architecture[1] to sandbox (AI-based) unverified controllers over stochastic games. The repository contains two folders: 
1. **Safe_visor_Architecture_for_Stochastic_Game:** Code for computing approximate probabilistic relations[3], synthesizing the correct-by-construction symbolic controllers for stochastic games[3], and constructing the corresponding Safe-visor architecture[1,2,4]. You are also referred to [5] and [6] for the high-level ideas regarding how the symbolic controllers and their corresponding Safe-visor architecture work.
2. **Repository for AAAI37 Papers:** Codes for the paper *Towards Safe AI: Sandboxing DNNs-based Controllers in Stochastic Games*, in Proceeding of the 37th AAAI Conference on Artificial Intelligence

### Citation
We are more than happy if you find the code useful for your research. If you use the code for 1) building the approximate probabilistic relations; or/and 2) synthesizing the symbolic controller, please cite the following paper:

    @article{zhong2023automata,
      title={Automata-based controller synthesis for stochastic systems: A game framework via approximate probabilistic relations},
      author={Zhong, Bingzhuo and Lavaei, Abolfazl and Zamani, Majid and Caccamo, Marco},
      journal={Automatica},
      volume={147},
      pages={110696},
      year={2023},
      publisher={Elsevier}
    }

If you are using the code for synthesizing the Safe-visor architecture, please cite the following paper:

    @InProceedings{Zhong2023Sandboxing,
      author        = {Bingzhuo Zhong and Hongpeng Cao and Majid Zamani and Marco Caccamo},
      booktitle     = {Proceedings of the Thirty-Seven AAAI Conference on Artificial Intelligence},
      title         = {Towards Safe AI: Sandboxing DNNs-based Controllers in Stochastic Games},
      year          = {2023},
    }

### Safe_visor_Architecture_for_Stochastic_Game
The following software/toolboxes for **MATLAB** are required for using the codes in this folder:
 1. Multi-Parametric Toolbox 3: https://www.mpt3.org/;
 2. YALMIP: https://yalmip.github.io/;
 3. MOSEK(Version 9.3.6): https://www.mosek.com/;
 4. MATLAB toolboxes including:
	 (1) Statistics and Machine Learning Toolbox (Version 11.6);
	 (2) Parallel Computing Toolbox (Version 7.1);
	 (3) MATLAB Parallel Server (Version 7.1);
	 (4) Polyspace Bug Finder (Version 3.1).
 
Next, we provide a brief introduction to how to use the files in this folder to synthesize synthesizing the correct-by-construction controller and the Safe-visor architecture in [4]. You can readily adapt these codes to your model after installing the software/toolboxes above.

### Repository for AAAI37 Paper:
Here is a brief introduction to synthesizing the Safe-visor architecture, training an AI-based agent, and simulating the results in [4].
#### Construction of Safe-visor architecture in the paper:  
To synthesize the Safe-visor architecture in MATLAB, you need to install the software mentioned above and add the folder "Safe_visor_Architecture_for_Stochastic_Game". The codes for synthesizing the Safe-visor Architecture $sva_E$ and $sva_N$ locate in the folders ***Quadrotor\_4dim\_A\_E*** and **Quadrotor\_4dim\_A\_N**, respectively. To synthesize $sva_E$, one should:
- Include folder ***Quadrotor\_4dim\_A\_E*** and its sub-folders in the path;
- Go to the folder ***Quadrotor\_4dim\_A\_E***;
- Run the m-file ***config\_Quadrotor\_4dim\_A\_E.m***.

After the execution is done, one obtains a mat-file ***Drone\_invariance\_sva.mat*** that contains the Safe-visor $sva_E$. To synthesize $sva_N$, one should:
- Include folder ***Quadrotor\_4dim\_A\_N*** and its sub-folders in the path;
- Go to the folder ***Quadrotor\_4dim\_A\_N***;
-  Run the m-file ***config\_Quadrotor\_4dim\_A\_N.m***.
After the execution is done, one obtains a mat-file ***Drone\_DFA\_sva.mat*** that contains the Safe-visor $sva_N$.
#### Analyzing the simulation for the case study in the paper: 
The codes for running simulation using the Safe-visor Architecture locate in the folders ***Quadrotor\_4dim\_simulation***. To simulate the case study, one should:
-  Include folder ***Quadrotor\_4dim\_simulation*** and its sub-folder in the path;
- Copy mat-files ***Drone\_invariance\_sva.mat*** and ***Drone\_DFA\_sva.mat***, which are obtained as in the previous step, to the folder ***Quadrotor\_4dim\_simulation***;
- Run Python script for DNNs inference: ***python ddpg\_controller.py***;
- Run the m-file ***config\_simulation\_Quadrotor\_4d\_sim.m***
After the simulation completes, one gets a mat-file ***eight\_monte\_data.mat*** that contains the simulation data.
Then, go to the folder ***analysing\_simulation\_data*** and run the m-file ***analyze\_sim\_Quadrotor.m*** to:
-  Compute the average acceptance rate, average execution time and the standard deviation of the execution time for the Safe-visor architecture;
- Export csv-file containing the simulation data for generating figures in the manuscript.
To plot the simulation experiment results, we run `python plot.py`. 


#### Training and testing of the DNNs agent
The training of the DNNs agent is done in python using TensorFlow framework. All codes for the training and
the testing can be found in the folder “Training DNNs controller”. To run the training, one should generate
configuration files first. One can also use the configuration file ( “./config/ddpg drn.json”) which is used
for the experiments in this paper. We define the main entrance for training and testing in the python script
“main ddpg.py”. And the training and testing can be executed via running the script “ python main ddpg.py”
with specified parameters (detail parameters can be found in the code). Examples are listed as follows:

- Generating new configuration file: 
`python main ddpg.py −−generate config`
- Running training: 
`python main ddpg.py −−config [path-to-config-files] −−mode train`
- Running testing: 
`python main ddpg.py −−config [path-to-config-files] −−mode test −−weights [path-to-pretrained model]`

One can use the pre-trained model provided for a quick testing. The pre-trained model can be found in the
folder ‘pretrained model’. To visualize the testing, one can change the visualization parameter in the running
command, for example:

- Running testing with pretrained model and visualization:
`python main ddpg.py −−config config/ddpg drn.json −−mode test −−weights pretrained model
−−params stats params/visualize eval true`


### References
[1]  **B. Zhong**, M. Zamani, and M. Caccamo,  [Sandboxing controllers for stochastic Cyber-Physical Systems](https://link.springer.com/chapter/10.1007/978-3-030-29662-9_15), In: 17th International Conference on Formal Modelling and Analysis of Timed Systems (FORMATS), Lecture Notes in Computer Science (VOL 11750), Springer, 2019.

[2] **B. Zhong**, A. Lavaei, H. Cao, M. Zamani, and M. Caccamo, [Safe-visor architecture for sandboxing (AI-based) unverified controllers in stochastic Cyber-Physical Systems](https://www.sciencedirect.com/science/article/pii/S1751570X2100100X?via%3Dihub), In: Nonlinear Analysis: Hybrid Systems, 43C, 2021

[3] **B. Zhong**, A. Lavaei, M. Zamani, and M. Caccamo, [Automata-based controller synthesis for stochastic systems: A game framework via approximate probabilistic relations](https://doi.org/10.1016/j.automatica.2022.110696), In: Automatica, 147C, 2023.

[4] **B. Zhong**, H. Cao, M. Zamani, and M. Caccamo, Towards Safe AI: Sandboxing DNNs-based Controllers in Stochastic Games, In Proceeding of the 37th AAAI Conference on Artificial Intelligence, to appear

[5]  **B. Zhong**, A. Lavaei, M. Zamani, and M. Caccamo,  [Poster abstract: Controller synthesis for nonlinear stochastic games via approximate probabilistic relations](https://doi.org/10.1145/3501710.3524732), In: Proceeding of 25th ACM International Conference on Hybrid Systems: Computation and Control (HSCC), ACM, 2022

[6] A. Lavaei,  **B. Zhong**, and M. Caccamo, and M. Zamani,  [Towards trustworthy AI: safe-visor architecture for uncertified controllers in stochastic Cyber-Physical Systems](https://dl.acm.org/doi/abs/10.1145/3457335.3461705), In: Proceedings of the International Workshop on Computation-Aware Algorithmic Design for Cyber-Physical Systems (CAADCPS), ACM, 2021

