#title run alphafold msa parallely

#@markdown Using TDC DAVIS dataset <br>
#@markdown https://tdcommons.ai/multi_pred_tasks/dti/ <br>
#@markdown 25,772 DTI pairs, 68 drugs, 379 proteins.

TOTAL_TARGET = 379
TARGET_DIR = '/data/project/MinGyu/input/alphafold/tdc/davis/targets.csv'

from threading import Thread
import subprocess
import pandas as pd

def extract_targets(DIR):
    f = open(DIR, 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
        
    return lines

def run(id, ID):
    print("QUERY : " + ID)
    print("##################")
    print(" ")
    subprocess.call("nohup ./run_alphafold.sh -d /data/project/MinGyu/af_download_data -o /data/project/MinGyu/output/alphafold/davis -p monomer_ptm -i /data/project/MinGyu/input/alphafold/tdc/davis/"+ ID + ".fasta -t 2021-07-27 -m model_1 -f", shell = True)
    return

if __name__ == "__main__":
    unique = extract_targets(TARGET_DIR)
    
    status = 0
    count = 0
    
    for ID in unique:
        print("##################")
        print("progress: " + str(int(status * 100 / TOTAL_TARGET)) + " % (" + str(status) + ' / ' + str(TOTAL_TARGET) + ')')
        status += 1
        count += 1
        if count == 1:
            th1 = Thread(target=run, args=(status, ID))
            th1.start()
            if status == TOTAL_TARGET:
                th1.join()
        elif count == 2:
            th2 = Thread(target=run, args=(status, ID))
            th2.start()
            if status == TOTAL_TARGET:
                th1.join()
                th2.join()
        elif count == 3:
            th3 = Thread(target=run, args=(status, ID))
            th3.start()
            if status == TOTAL_TARGET:
                th1.join()
                th2.join()
                th3.join()

        elif count == 4:
            th4 = Thread(target=run, args=(status, ID))
            th4.start()
            if status == TOTAL_TARGET:
                th1.join()
                th2.join()
                th3.join()
                th4.join()

        elif count == 5:
            th5 = Thread(target=run, args=(status, ID))
            th5.start()
            if status == TOTAL_TARGET:
                th1.join()
                th2.join()
                th3.join()
                th4.join()
                th5.join()
        else:
            th6 = Thread(target=run, args=(status, ID))
            th6.start()

            th1.join()
            th2.join()
            th3.join()
            th4.join()
            th5.join()
            th6.join()
            count = 0
print("parallel computing finished")
