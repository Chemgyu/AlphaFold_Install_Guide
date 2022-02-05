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

---

## 3. Package Setup

Main reference: [non_docker github](https://www.notion.so/AlphaFold2-Setup-Guide-1fc268d4deaf4108b9aa368f2f6dc585).

- Conda environment adoption & activation
    
    ```bash
    conda create --name alphafold python==3.8
    conda activate alphafold
    ```
    
- Install basic dependencies
    
    ```
    conda install -y -c conda-forge openmm==7.5.1 cudnn==8.2.1.32 cudatoolkit==11.0.3 pdbfixer==1.7
    conda install -y -c bioconda hmmer==3.3.2 hhsuite==3.3.0 kalign2==2.04
    ```
    
- Clone github repositories
    
    ```bash
    git clone https://github.com/deepmind/alphafold.git
    git clone https://github.com/kalininalab/alphafold_non_docker.git
    git clone https://github.com/Zuricho/ParallelFold.git
    ```
    
- Download chemical properties
    
    ```bash
    wget -P /data3/project/MinGyu/ParallelFold/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt --no-check-certificate
    ```
    
- Install other dependencies
    
    ```bash
    pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 scipy==1.7.0 tensorflow==2.5.0 pandas==1.3.4 tensorflow-cpu==2.5.0
    pip install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
    pip install jaxlib==0.1.75
    ```
    
- Apply openMM patch
    
    ```bash
    cd ~/.conda/envs/alphafold/lib/python3.8/site-packages/ && patch -p0 < /data/project/MinGyu/ParallelFold/docker/openmm.patch
    ## OUTPUT ##
    #  patching file simtk/openmm/app/topology.py
    #  Hunk #1 succeeded at 353 (offset -3 lines).
    ```
