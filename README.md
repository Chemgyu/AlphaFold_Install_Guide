# AlphaFold_Install_Guide
Hands-on installation guide of AlphaFold for non-CS protein researchers

> Created by: **MinGyu Choi** <br>
Last modified: Jan14 2022
Contact: chemgyu98@snu.ac.kr | aigengyu@gmail.com

> This repository deals with three methods for using AlphaFold2 in both ***local environment*** and ***cloud-computing environment*** (`Amazon cloud` and `Google colab`).

> You can easily build your protein structure prediction environment by reproducing following instructions in this page.

> Please feel free to contact me if you faced an error or if you have any suggestions.


### 0. Overview / References
1. Main References
    1) Atricle
        Jumper2021, the original alphafold2 paper [Link](https://www.nature.com/articles/s41586-021-03819-2.pdf)
        Zhong2021, for msa / alphafold separation [Link](https://arxiv.org/pdf/2111.06340.pdf)
    2) Github
        AlphaFold official [Link](https://github.com/deepmind/alphafold)
        Non-docker setup [Link](https://github.com/kalininalab/alphafold_non_docker)
        msa/alphafold separation [Link1](https://github.com/Zuricho/ParallelFold) [Link2](https://github.com/SJTU-HPC/ParaFold)
    3) AWS Related
        Amazon blog [Link](https://aws.amazon.com/ko/blogs/machine-learning/run-alphafold-v2-0-on-amazon-ec2/)
        Medium article [Link](https://medium.com/proteinqure/alphafold-quickstart-on-aws-9ba20692c98e)
    4) Google Colab
2. Non-Docker Setup
3. CPU/GPU Separation
4. File Structure


### 1. Google Colab AlphaFold2
Main Reference: [Official AlphaFold Colab](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb)
