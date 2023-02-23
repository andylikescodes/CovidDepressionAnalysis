import sys
import os


output_path = "/home/deliang@chapman.edu/Documents/CovidDepressionAnalysis/output/cluster"

out_path = "--output={}/impute_cvd_cases.out ".format(output_path)
err_path = "--error={}/impute_cvd_cases.err ".format(output_path)
job_name = "--job-name=KNN_impute_cvd_cases "
parameters = "--export=K=3,percentage_missing=10 "

command = "sbatch " + out_path + err_path + job_name + parameters + 'impute_cvd_cases.sbatch'

print(command)

os.system(command=command)
        