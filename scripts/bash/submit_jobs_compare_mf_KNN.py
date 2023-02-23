import sys
import os


output_path = "/home/deliang@chapman.edu/Documents/CovidDepressionAnalysis/output/cluster"

out_path = "--output={}/compare_mf_knn.out ".format(output_path)
err_path = "--error={}/compare_mf_knn.out ".format(output_path)
job_name = "--job-name=compare_mf_knn "
parameters = "--export=mf_K1=10,mf_K2=5,knn_K1=5,knn_K2=1,iter=1500,percentage_missing=10 "

command = "sbatch " + out_path + err_path + job_name + parameters + 'impute_core_vars.sbatch'

print(command)

os.system(command=command)
        