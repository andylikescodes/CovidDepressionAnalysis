import sys
import os

k1 = [8, 9, 10]
k2 = [1, 2, 3, 4, 5]

# k1 = [1]
# k2 = [1]

output_path = "/home/deliang@chapman.edu/Documents/CovidDepressionAnalysis/output/cluster"

for i in k1:
    for j in k2:
        out_path = "--output={}/impute_k1={}_k2={}.out ".format(output_path, str(i), str(j))
        err_path = "--error={}/impute_k1={}_k2={}.out ".format(output_path, str(i), str(j))
        job_name = "--job-name=k1={}_k2={} ".format(str(i), str(j))
        parameters = "--export=K1={},K2={} ".format(str(i), str(j))
        
        command = "sbatch " + out_path + err_path + job_name + parameters + 'impute.sbatch'
        
        print(command)
        
        os.system(command=command)
        
        
        

