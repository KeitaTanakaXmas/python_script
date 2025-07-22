import subprocess
import sys
import os
from Terminal_Log import execute_shell_script, time_now, create_directory


def run_searchflickpix(infile:str, base_outfile:str, logprob2_values:list, n_division_values:list):
    """
    Execute the searchflickpix command with the given parameters.

    Args:
        infile (str): The input file name.
        base_outfile (str): The base output file name.
        logprob2_values (list): A list of logprob2 values to iterate over.
        n_division_values (list): A list of n_division values to iterate over.

    Returns:
        None
    """
    time, date = time_now(False)
    print('--------------------------------------')
    print('Multi_SFP.py')
    print(f'infile = {infile}')
    print(f'base_outfile = {base_outfile}')
    print(f'logprob2_values = {logprob2_values}')
    print(f'n_division_values = {n_division_values}')
    print('--------------------------------------')
    create_directory('Log')
    for logprob2 in logprob2_values:
        print(f'logprob2 = {logprob2}')
        for n_division in n_division_values:
            print(f'n_division = {n_division}')
            outfile  = f"{base_outfile}_{n_division}_{str(logprob2).replace('.', '')}.evt"
            outfile2 = f"{base_outfile}_{n_division}_{str(logprob2).replace('.', '')}.fpix"
            cmd_fpix = f"searchflickpix infile={infile} outfile={outfile} cellsize=5 grade=0,1,2,3,4,5,6 impfac=320 logprob1=10 logprob2={logprob2} iterate=no n_division={n_division} bthresh=1"
            cmd_evt  = f"searchflickpix infile={infile} outfile={outfile2} cellsize=5 grade=0,1,2,3,4,5,6 impfac=320 logprob1=10 logprob2={logprob2} iterate=no n_division={n_division} bthresh=1 cleanimg=yes"
            print('Processing:', cmd_fpix)
            execute_shell_script(cmd_fpix,f'./Log/run_searchflickpix_{n_division}_{logprob2}_fpix_{date}_{time}.log')
            print('Processing:', cmd_evt)
            execute_shell_script(cmd_evt,f'./Log/run_searchflickpix_{n_division}_{logprob2}_evt_{date}_{time}.log')

if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else "xtend_exclude_calsource.evt"
    base_outfile = sys.argv[2] if len(sys.argv) > 2 else "SFP_trial"
    logprob2_values = [-4,-10,-20,-30]
    n_division_values = [1,2]

    run_searchflickpix(infile, base_outfile, logprob2_values, n_division_values)


