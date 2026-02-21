# Adaptive ECN Thresholds in DCNs with Reinforcement Learning using Spiking Neural Networks (SNNs)  
This repository contains the forked High-Precision-Congestion-Control repository (which can be found [here](https://github.com/alibaba-edu/High-Precision-Congestion-Control)) which contains the implementations for the congestion control algorithms, such as DCQCN and the state-of-the-art HPCC.  

I have modified/added the following files to add the adaptive ECN threshold functionality:
- simulation/run.py (modified to parse an 'adaptive ECN marking' argument)
- simulation/scratch/third.cc (modified to add the reinforcement learning functionality needed for the simulation)
- simulation/scratch/rl-agent.py (the reinforcement learning agent)

Note: Other files may have been changed, all changes (in files that weren't created by me - which will be listed above) can be found by searching the directory for 'Robert Lucas' as I've added identifying comments around my changes.  

## Steps to run this code  
- Clone the NS3-AI repository (v1.1.0) into the 'simulation/src' directory, which the tar file can be found [here](https://github.com/hust-diangroup/ns3-ai/archive/refs/tags/v1.1.0.tar.gz). Note: version 1.1.0 must be used as this is the last version that use the waf build system rather than CMake.  
- Create two anaconda environments, one with python3 and one with python2.
