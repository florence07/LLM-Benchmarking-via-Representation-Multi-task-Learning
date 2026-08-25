Unequal-J T=18, R=100 local run
================================

Requirements
------------
Python 3.10 or later

Setting
-------
N = 100
T = 18
J_t = [50, 60, 70, 80, 90, 100] repeated 3 times
H_VALUES = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
C_lambda = 0.1, 0.2, ..., 2.0
R = 100
CV_FOLDS = 5
CV_JOBS = 8
BASE_SEED = 20260819

macOS or Linux
--------------
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_unequalJ_T18_R100_local.py

Windows PowerShell
------------------
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_unequalJ_T18_R100_local.py

The full run contains 1000 datasets. Results are saved after every dataset in
results_unequalJ_T18_R100. Running the same command again skips completed tasks.
When all tasks are complete, the combined CSV files are created automatically
in results_unequalJ_T18_R100/combined.

Pilot run
---------
python run_unequalJ_T18_R100_local.py --start-task 0 --end-task 0

Rerun the selected task
-----------------------
python run_unequalJ_T18_R100_local.py --start-task 0 --end-task 0 --force

Combine manually
----------------
python combine_unequalJ_T18_R100.py
