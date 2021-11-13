# KosickaKosacka


## Install MiniConda
    https://docs.conda.io/en/latest/miniconda.html

## Install Python 3.9
Open Anaconda Powershell Prompt:

    conda create -n tf_test_env1 python=3.9

## Activate Virtual Environment (Dependencies folder) (FIRST SETUP OF EXISTING PROJECT)

    conda activate tf_test_env1

## Install App Dependencies for requirements file ) (FIRST SETUP OF EXISTING PROJECT
Open Anaconda Powershell Prompt:

    pip install -r requirements.txt --user

## Run algo version
    
    python main.py --maps Greenland Localhost 'Mapname with spaces' --base_url http://169.51.194.78:31798/

## Run algo version with Rendering Mode
    
    python main.py --render_mode True --maps Greenland Localhost 'Mapname with spaces' --base_url http://169.51.194.78:31798/


## Run NN verision
### Run Training Mode

    python main_dqn.py
    
### Run Training Mode with Rendering Mode

    python main_dqn.py --render_mode True --maps Greenland Localhost 'Mapname with spaces' --base_url http://169.51.194.78:31798/

### Run Predict Mode

    python main_dqn.py --predict_mode True --maps Greenland Localhost 'Mapname with spaces' --base_url http://169.51.194.78:31798/

### Run Predict Mode with Rendering Mode

    python main_dqn.py --predict_mode True --render_mode True --maps Greenland Localhost 'Mapname with spaces' --base_url http://169.51.194.78:31798/


###
    conda activate C:\Users\majof\anaconda3\envs\tf_test_env2
    cd ../../IBM/Hack@Home2021/KosickaKosacka
