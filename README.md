# AlphaFold_Install_Guide
Hands-on installation guide of AlphaFold for non-CS protein researchers

> Created by: **MinGyu Choi** <br>
Last modified: Jan14 2022
Contact: chemgyu98@snu.ac.kr | aigengyu@gmail.com

> This repository deals with three methods for using AlphaFold2 in both ***local environment*** and ***cloud-computing environment*** (`Amazon cloud` and `Google colab`).

> You can easily build your protein structure prediction environment by reproducing following instructions in this page.

> Please feel free to contact me if you faced an error or if you have any suggestions.

(update Feb5) You can also use this page using `notion` application [NOTION_LINK](https://helpful-longan-258.notion.site/AlphaFold2-Setup-Guide-1fc268d4deaf4108b9aa368f2f6dc585)


## 0. Overview / References
### 1. Main References
1) Atricle <br>
    - Jumper2021, the original alphafold2 paper [Link](https://www.nature.com/articles/s41586-021-03819-2.pdf) <br>
    - Zhong2021, for msa / alphafold separation [Link](https://arxiv.org/pdf/2111.06340.pdf) <br>
2) Github <br>
    - AlphaFold official [Link](https://github.com/deepmind/alphafold) <br>
    - Non-docker setup [Link](https://github.com/kalininalab/alphafold_non_docker) <br>
    - msa/alphafold separation [Link1](https://github.com/Zuricho/ParallelFold) [Link2](https://github.com/SJTU-HPC/ParaFold) <br>
3) AWS Related <br>
    - Amazon blog [Link](https://aws.amazon.com/ko/blogs/machine-learning/run-alphafold-v2-0-on-amazon-ec2/) <br>
    - Medium article [Link](https://medium.com/proteinqure/alphafold-quickstart-on-aws-9ba20692c98e) <br>
4) Google Colab <br>
    - AlphaFold official [Link](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb) <br>
    - ColabFold (Steinegger M. et al.) [Link](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb) <br><br>
### 2. Non-Docker Setup [Link](https://github.com/kalininalab/alphafold_non_docker) <br><br>
### 3. CPU/GPU Separation [Link1](https://github.com/Zuricho/ParallelFold) [Link2](https://github.com/SJTU-HPC/ParaFold) <br>
<br><br>

## 1. Google Colab AlphaFold2
Main Reference: [Official AlphaFold Colab](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb) <br>
Follow instructions inside the colab page.
Main reference: [Official AlphaFold Colab](https://www.notion.so/AlphaFold2-Setup-Guide-1fc268d4deaf4108b9aa368f2f6dc585)

- (optional) Extracting single/pair representations
    - Modify `GIT_REPO` URL
        - Fork AlphaFold to your github repository
        - Edit `module.py` for returning representations
            
            `alphafold/model/modules.py` line 270
            
            ```bash
            class AlphaFold(hk.Module):
            
              def __init__(self, config, name='alphafold'):
                super().__init__(name=name)
                self.config = config
                self.global_config = config.global_config
            
              def __call__(
                  self,
                  batch,
                  is_training,
                  compute_loss=False,
                  ensemble_representations=False,
                  return_representations=True):
            			# it was return_representations=False 
            ```
            
    - Modify Colab Code.
        
        ```python
        # **Before**
        GIT_REPO = 'https://github.com/deepmind/alphafold'
        
        # **After** (Example, insert your github repository URL.)
        GIT_REPO = 'https://github.com/Chemgyu/alphafold'
        ```
---

## 2. AWS Server Setup

Main reference: [AWS blog](https://www.notion.so/AlphaFold2-Setup-Guide-1fc268d4deaf4108b9aa368f2f6dc585)

- AWS EC2 Instance setup
    - AWS EC2 console → choose AWS region (top right, Asia - Seoul)
    - Choose new EC2 instance → Search Deep Learning AMI → Latest Ubuntu server
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e2e05d35-2b62-4c46-af94-e17f2d65ed1b/Untitled.png)
        
    - Select the c6i.large instance
        
        <aside>
        💡 Note: 
        downloading alphafold database requires > 10 hr running time
        ⇒ use the cheapest instance type for downloading.
        (I used `c6i.large` instance)
        
        </aside>
        
    - Storage volume setting
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/844834ec-505d-43d9-a12c-8fa53e372fe8/Untitled.png)
        
    - Allow ssh and download .pem key for convenient access
- Disk setup
    - Update pre-installed packages
        
        ```bash
        sudo apt update
        ```
        
    - Disk status check
        
        [Make an Amazon EBS volume available for use on Linux](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html)
