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

---

## 4. Download af_download_data

Main reference: [non_docker github](https://www.notion.so/AlphaFold2-Setup-Guide-1fc268d4deaf4108b9aa368f2f6dc585).

Use `*alphafold_non_docker/download_db.sh*`

- Modify `download_db.sh` for fast download
    - mkdir issue: remove every `mkdir ~` arguments after first run
    - For continuous download: add `-c` argument for every wget argument
    - For parallel download: Separate each download arguments in separate files
        
        ```bash
        EXAMPLE)
        download_dbs.sh    #< This takes huge amount of time!
        download_mmcif.sh  #< This also takes long time.
        download_mgnify.sh
        download_uniprot.sh #<
        download_others.sh #< Other files are not such big.
        ```
                
- Give permission to bash files
    
    ```bash
    chmod +x download_dbs.sh
    chmod +x download_mmcif.sh
    chmod +x download_mgnify.sh
    chmod +x download_uniprot.sh
    chmod +x download_others.sh
    ```
    
- Run each bash files
    
    ```bash
    # redirected output logs for separate .out files for convenience.
    nohup bash download_dbs.sh -d /data/af_download_data > dbs_download.out &
    nohup bash download_mmcif.sh -d /data/af_download_data > mmcif_download.out &
    nohup bash download_mgnify.sh -d /data/af_download_data > mgnify_download.out &
    nohup bash download_uniprot.sh -d /data/af_download_data > uniprot_download.out &
    nohup bash download_others.sh -d /data/af_download_data > others_download.out &
    ```
    
- If connection lost: re-run aborted download
- Progress check: use `tail -f` command. (toggle: codes)
    
    ```bash
    tail -f dbs_download.out
    tail -f mmcif_download.out
    tail -f mgnify_download.out
    tail -f uniprot_download.out
    tail -f others_download.out
    ```
    
- Make `pdb70` directory
    
    ```bash
    # @ /data/project/MinGyu
    mkdir /data/project/MinGyu/af_download_data/pdb70/pdb70
    ```

---

## 5. Run CPU part (MSA)

Main reference: [ParaFold github](https://www.notion.so/AlphaFold2-Setup-Guide-1fc268d4deaf4108b9aa368f2f6dc585).

Use `ParallelFold/run_alphafold.sh`

- Modify `tmp` directory
    - modify `utils.py`
        
        Edit File:  `ParallelFold/alphafold/data/tools/utils.py`
        
        ```bash
        # **BEFORE** (LINE 25)
        def tmpdir_manager(base_dir: Optional[str] = None):
        #^ This will save tmp files at /tmp, which is small for AlphaFold2
        
        # **AFTER** 
        def tmpdir_manager(base_dir: Optional[str] = "/data/mnt/alphafold/tmp"):
        ```
        
    - `mkdir /data/mnt/alphafold/tmp` directory
    - Sample error message without modification
        
        ```bash
        File "/data/ParallelFold/alphafold/data/tools/jackhmmer.py", line 139, in _query_chunk
            raise RuntimeError(
        RuntimeError: Jackhmmer failed
        stderr:
        Fatal exception (source file esl_msafile_stockholm.c, line 1249):
        stockholm msa write failed
        system error: No space left on device
        ```
        
    
    [https://github.com/deepmind/alphafold/issues/280](https://github.com/deepmind/alphafold/issues/280)
    
- Routine for fresh running
    - Conda Activation
        
        ```bash
        conda activate alphafold
        ```
        
    - (AWS) Disk Mount
        
        ```bash
        sudo mount /dev/nvme1n1 /data
        ```
        
    - (AWS) Swap Expansion
        
        ```bash
        sudo dd if=/dev/zero of=/data/swp/swapfile50g bs=1024 count=52428800
        sudo mkswap /data/swp/swapfile50g
        swapon /root/swp/swapfile50g
        ```
        
        swap file is used as an alternative RAM, which is actually on the disk.
        
        at parallel computing (2 threads), requires ~40G swap capacity.
        
        - Error message
            
            ```bash
            E0109 16:32:18.239599 140375667808064 hhblits.py:138] HHblits failed. HHblits stderr begin:
            E0109 16:32:18.239672 140375667808064 hhblits.py:142] HHblits stderr end
            Traceback (most recent call last):
              File "/data/ParallelFold/run_alphafold.py", line 455, in <module>
                app.run(main)
              File "/home/ubuntu/anaconda3/envs/alphafold/lib/python3.8/site-packages/absl/app.py", line 312, in run
                _run_main(main, args)
              File "/home/ubuntu/anaconda3/envs/alphafold/lib/python3.8/site-packages/absl/app.py", line 258, in _run_main
                sys.exit(main(argv))
              File "/data/ParallelFold/run_alphafold.py", line 429, in main
                predict_structure(
              File "/data/ParallelFold/run_alphafold.py", line 175, in predict_structure
                feature_dict = data_pipeline.process(
              File "/data/ParallelFold/alphafold/data/pipeline.py", line 205, in process
                hhblits_bfd_uniclust_result = run_msa_tool(
              File "/data/ParallelFold/alphafold/data/pipeline.py", line 97, in run_msa_tool
                result = msa_runner.query(input_fasta_path)[0]
              File "/data/ParallelFold/alphafold/data/tools/hhblits.py", line 143, in query
                raise RuntimeError('HHblits failed\nstdout:\n%s\n\nstderr:\n%s\n' % (
            RuntimeError: HHblits failed
            stdout:
            
            stderr:
            ```
            
        
        [https://github.com/kalininalab/alphafold_non_docker/issues/12](https://github.com/kalininalab/alphafold_non_docker/issues/12)
        
- Run MSA
    
    ```bash
    nohup ./run_alphafold.sh -d /data/project/MinGyu/af_download_data -o /data/project/MinGyu/output/alphafold -p monomer_ptm -i /data/project/MinGyu/input/alphafold/test.fasta -t 2021-07-27 -m model_1 -f
    # please carefully read an required argument description at run_alphafold.sh bash file.
    ```
    
- About parallel computing
    - HHblits and other msa methods have to do a lot of I/O operations
        
        ⇒ Do not run alphafold in multiprocess manner unless you can use very fast SSD.
        
        [https://github.com/deepmind/alphafold/issues/123](https://github.com/deepmind/alphafold/issues/123)
        
    - You can maximize throughput by introducing multithreading concept
        - I attached a naive python code for 6-way multithreading. See `multithreading/run_alphafold.py`
        
            
- Sample Results
    
    ```bash
    test
    ├── features.pkl
    ├── msas
    │   ├── bfd_uniclust_hits.a3m.   # ~ 2 hr for kinases
    │   ├── mgnify_hits.sto          # ~ 1000 sec for kinases
    │   ├── pdb_hits.hhr             # ~ 200 sec for kinases
    │   └── uniref90_hits.sto        # ~ 1000 sec for kinases
    #├── result_model_1.pkl         #< result after structure prediction
    #└── unrelaxed_model_1.pdb      #< result after structure prediction
    ```
    
- Tips
    
    <aside>
    💡 Because of FILE I/O, alphafold2 spends a lot of  time (up to 3 hr for one query)
    
    </aside>
    
    <aside>
    💡 Most of error arise while executing HHblits (resulting bfd_uniclust_hits.a3m)
    ⇒ At error case, you should re-run the whole msa now.
    
    </aside>

---

## 6. Run GPU part (evoformer + structure module)

Main reference: [ParaFold github](https://www.notion.so/AlphaFold2-Setup-Guide-1fc268d4deaf4108b9aa368f2f6dc585).

Note: Also can be executed on CPU environment

- (optional) Edit `module.py` for extracting representations
    
    `/data/ParallelFold/alphafold/model/modules.py` line 270
    
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
    
- Run structure prediction
    
    ```bash
    nohup ./run_alphafold.sh -d /data/project/MinGyu/af_download_data -o /data/project/MinGyu/output/alphafold -p monomer_ptm -i /data/project/MinGyu/input/alphafold/test.fasta -t 2021-07-27 -m model_1 &
    ```
    
- Sample Results
    
    ```bash
    test
    ├── features.pkl
    ├── msas
    │   ├── bfd_uniclust_hits.a3m
    │   ├── mgnify_hits.sto
    │   ├── pdb_hits.hhr
    │   └── uniref90_hits.sto
    ├── result_model_1.pkl         #< result after structure prediction
    └── unrelaxed_model_1.pdb      #< result after structure prediction
    ```
    
- (optional) Tips for extracting representations
    
    You can access MSA, single repr., pair repr., and other useful vector data by using `pickle` package and `result_model_1.pkl`.
