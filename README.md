# Project 3: Continuous Control

## 1.- Introduction

This repository includes the code of my personal attempt to solve the third work project of the Udacity Nanodegree.
In the following photograph it is posible to see a representation of the environment solved with two trained agents playing tennis.

**FIGURA**



## 2. First steps in the environment

This environment has been built using the **Unity Machine Learning Agents Toolkit (ML-Agents)**, which is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. You can read more about ML-Agents by perusing this [GitHub repository](https://github.com/Unity-Technologies/ml-agents).  

The project environment provided by Udacity is similar to, but not identical to the Tennis environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

> Is is required to work with the environment provided by Udacity platform as part of the project.


In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of `+0.1`. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of `-0.01`. Thus, the goal of each agent is to keep the ball in play.

#### 2.1. State and action spaces

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

#### 2.2. Solving the environment
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a **single score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least **+0.5.**


## 3. Included in this repository

* The code used to create and train the Agent
  * Tennis.ipynb
  * ddpg_agent.py
  * model.py
* The weights of the trained model
  * checkpoint_actor1.pth
  * checkpoint_critic1.pth
  * checkpoint_actor2.pth
  * checkpoint_critic2.pth
* A Report.md file describing the development process and the learning algorithm, along with ideas for future work
* This README.md file

## 4. Setting up the environment

This section describes how to get the code for this project, configure the local environment, and download the Unity environment with the Agents.

#### 4.1 Getting the code
You have two options to get the code contained in this repository:
##### Option 1. Download it as a zip file

* [Click here](https://github.com/ASO92/DRL_p3_CollaborationCompetition_Udacity/archive/master.zip) to download all the content of this repository as a zip file
* Decompress the downloaded file into a folder of your choice


##### Option 2. Clone this repository using Git version control system
If you are not sure about having Git installed in your system, run the following command to verify that:
```
$ git --version
```
Having Git installed in your system, you can clone this repository by running the following command:

```
$ git clone https://github.com/ASO92/DRL_p2_ContinuousControl_Udacity/archive.git
```
#### 4.2 Installing Ananconda/Miniconda
Miniconda is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib, and a few others.  

If you would like to know more about Anaconda, visit [this link](https://www.anaconda.com/).

In the following links, you find all the information to install **Miniconda** (*recommended*)

* Download the installer: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* Installation Guide: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

Alternatively, you can install the complete Anaconda Platform

* Download the installer: [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)
* Installation Guide: [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)

####  4.3 Configuring the local environment

#### Installing the packages
The following steps will guide you to install the necesary dependences to execute correctly the environment.

**1. Create the environment**  
```
$ conda create --name drlnd-p2-control python=3.6
$ conda activate drlnd-p2-control
```  

**2. Install PyTorch**  
Follow [this link](https://pytorch.org/get-started/locally) to select the right command for your system.  
Here, there are some examples which you can use, if it fit in your system:  

**a.** Linux or Windows

```
## Run this command if you have an NVIDIA graphic card and want to use it  
$ conda install pytorch cudatoolkit=10.1 -c pytorch

## Run this command, otherwise
$ conda install pytorch cpuonly -c pytorch
```  

**b.** Mac  
MacOS Binaries do not support CUDA, install from source if CUDA is needed

```
$ conda install pytorch -c pytorch  
```  


**3. Install Unity Agents**  

```
$ pip install unityagents
```  

#### 4.4. Download the Unity environment with the Agents  

For this project, you will not need to install Unity - you can download it from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## 5. Train the agent

To see the steps on how is trained the agent it is recommended to explore the [Tennis.ipynb](./DRL_p2_ContinuousControl_Udacity/Continuous_Control.ipynb) file and follow the isntructions on it.

And to see the process followed and the results obtained, please see the:


#### 5.2 Adjusting the Hyperparameters

## 6. Uninstall
If you wish to revert all the modifies in your system, and remove all the code, dependencies and programs installed in the steps above, you will want to follow the next steps.

#### 6.1 Uninstall Miniconda or Anaconda
To do so, please refer to [this link](https://docs.anaconda.com/anaconda/install/uninstall/).


#### 6.2 Remove the code
Simply delete the entire folder containing the code you downloaded in the step "Getting the code"
