# Deep Dueling On-Policy Learning

In Intelligent Transportation Systems (ITS), Deep Reinforcement Learning (DRL) is being rapidly adopted in many complex environments due to its ability to leverage neural networks to learn good strategies. A centralized TSC controller with a deep RL agent (DRL-agent) is trained by a novel [deep dueling on-policy learning method]() referred to as 2DSARSA.

<p align="center">
  <img src="img/DRL_Env_Model.png" alt="A Deep Reinforcement Learning Environment Model" width=60%">
</p>


## Motivation and Contributions

In Traffic Signal Control (TSC), there has been a number of research efforts to apply ML techniques in general and Reinforcement Learning (RL) in particular, to optimize TSC. However, there is insufficient study on the comparison between deep on-policy (DSARSA) and deep off-policy (DQN) in the context of TSC. This work takes the first step in addressing the important gap in the learning performance in these two fundamental approaches for TSC researchers. Our preliminary work have shown that DQN and 3DQN perform unstably in a complex environment when the state and action space are extremely large. To address this issue, overall, this work makes three significant contributions on the important and challenging topic of applying RL in TSC, namely, 1) a first comparison of two fundamental deep reinforcement learning approaches namely, on-policy learning and off-policy learning, 2) a novel way of representing the state of the environment using traffic flow maps, and 3) an intuitive yet novel rewards function using the power metric that co-optimizes the network throughput and the end-to-end delay. We elaborate on these in the following paragraphs.
 

## The Proposed DRL Agent

* Deep Dueling SARSA (2DSARSA) ([code](./code), [paper]())

<p align="center">
  <img src="img/DRL_Arc.png" alt="Deep Reinforcement Learning Architecture" width=80%">
</p>

One contribution of this work relates to the design reinforcement learning algorithm. As mentioned before, deep RL methods that use neural networks, can be broadly classified into off-policy and on-policy methods. We have proposed and designed a novel on-policy deep RL agent for a centralized controller for a network of TSC that incorporate [traffic flow maps]() as the state description, the [power metric based reward function](), [Dueling Neural Network Architectures](https://arxiv.org/abs/1511.06581), and [Experience Replay Memory](https://www.nature.com/articles/nature14236?wm=book_wap_0005) to improve traffic signal control. The results show that the RL agent can better understand an environment, effectively learn from environmental feedback through the reward function and the learning process converges faster than many existing algorithms. In addition, the RL agent outperforms traditional BP-based algorithms and well-known deep off-policy based RL agents.


## Traffic Flow Maps (TFMs)

Another fundamental problem in applying RL to a network of traffic intersections is the state space explosion problem. The state space grows exponentially both with increase in the fidelity of the state description as well as with the number of traffic intersections. An important contribution of this paper is to address this problem by describing the state using TFMs. In TFM, the state variables which can be real numbers (such as the waiting time of the Head-of-Line vehicle) are mapped into color map to transform the state into an image. This allows the state to be described with arbitrary high fidelity and capture and store dynamic traffic flows for a network of multiple intersections. We propose TFMs that capture head-of-the-line (HOL) sojourn times for traffic lanes and HOL differences for adjacent intersections.

<p align="center">
<img src="img/TFM_One_with_4_movements.png" alt="TFM for One Intersection with 4 Movements" width=36%/> 
<br>
<br>
<br>
<img src="img/455.png" alt="TFM_455s" width=32%/> 
<img src="img/1155.png" alt="TFM_1155s" width=32%/> 
<img src="img/1755.png" alt="TFM_1755s" width=32%/> 
</p>

## Power Metric based Reward Function

The third critical aspect in any RL based approach is defining an appropriate reward function. This not only influences the learning performance but what the RL agent learns to  optimize. We proposed a novel reward function based on the power metric which is defined as the ratio of the system throughput to the end-to-end delay. In computer networks, this is referred to as the Klienrock's optimal operating point and is the basis of recent congestion control algorithm developed for the Internet. Based on detailed simulation analysis, it shown that the RL agent not only achieves good learning performance (faster convergence) but also achieve better performance in terms of network throughput and end-to-end delay.

<p align="center">
<img src="img/2DSARSA_Avg_EndToEnd_Delay.png" alt="Power Metric as Reward Function" width=32%/> 
<img src="img/2DSARSA_Avg_EndToEnd_Delay_R_throughput.png" alt="System Throughput as Reward Function" width=32%/> 
<img src="img/2DSARSA_Avg_EndToEnd_Delay_R_delay.png" alt="End-toEnd Delay as Reward Function" width=32%/> 
<br>
<br>
<img src="img/2DSARSA_Avg_Queue.png" alt="Power Metric as Reward Function" width=32%/> 
<img src="img/2DSARSA_Avg_Queue_R_throughput.png" alt="System Throughput as Reward Function" width=32%/> 
<img src="img/2DSARSA_Avg_Queue_R_delay.png" alt="End-toEnd Delay as Reward Function" width=32%/> 
</p>

## Learning Performance of The Proposed 2DSARSA

The 2DSARSA agent is able to perform remarkably well in a complicated traffic network of multiple intersections in our [work](). We have shown that the proposed 2DSARSA architecture has a significantly better learning performance compared to other DRL architectures including [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236), [3DQN](https://ieeexplore.ieee.org/document/8600382) and [Deep SARSA (DSARSA)](https://ieeexplore.ieee.org/abstract/document/7849837).

<p align="center">
<img src="img/Learning_Performance_2DSARSA.png" alt="Learning Performance of The Proposed 2DSARSA" width=70%/> 
</p>


## Environment Settings

* Set up your platform before running the code

### Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh or sha256sum Miniconda2-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
``` 

### Conda Create Environment
```bash
conda create --name env_name
conda activate env_name
``` 

### Install Tools

Add channel conda-forge
```bash
conda config --add channels conda-forge
``` 
Add channel pytorch
```bash
conda config --add channels pytorch
``` 

Install numpy
```bash
conda install numpy
``` 

Install pytorch with a specific version
```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
``` 

Install matplotlib
```bash
conda install matplotlib
``` 

Install scipy
```bash
conda install scipy
``` 

Install python-utils
```bash
conda install python-utils
``` 

Install utils
```bash
pip install utils
``` 

Install gym
```bash
pip install gym
``` 

Install ipython
```bash
conda install ipython
``` 

Install opencv
```bash
conda install opencv
``` 

Install opencv-python
```bash
pip install opencv-python
``` 

## Getting Started

You can clone this repository by:
```bash
git clone https://github.com/colouryen/2DSARSA.git
``` 

Switch to the code folder
```bash
cd 2DSARSA/code
``` 

Run the code
```bash
python RL_comp_multi.py
``` 

### Train a Model

You can readily train a new model for a traffic network of 9 intersections by running the [RL_comp_multi.py](./code/RL_comp_multi.py) script from the agent's main directory. 


### Save Model 

You can save a trained model in the ```saved_agents``` folder after running the [RL_comp_multi.py](./code/RL_comp_multi.py) script from the main folder.


## Citation

If you find this open-source release useful, please reference in your paper:

```
@inproceedings{yen2020deep,
  title={A Deep On-Policy Learning Agent for Traffic Signal Control of Multiple Intersections},
  author={Yen, Chia-Cheng and Ghosal, Dipak and Zhang, Michael and Chuah, Chen-Nee},
  booktitle={2020 IEEE Intelligent Transportation Systems Conference (ITSC)},
  pages={--},
  year={2020},
  organization={IEEE}
}
```
