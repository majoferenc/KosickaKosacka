# KosickaKosacka


## Install MiniConda
    https://docs.conda.io/en/latest/miniconda.html

## Install Python 3.9
Open Anaconda Powershell Prompt:

    conda create -n tf_test_env1 python=3.9

## Activate Virtual Environment (Dependencies folder) (FIRST SETUP OF EXISTING PROJECT)

    conda activate tf_test_env1

## Install App Dependencies for requirements file ) (FIRST SETUP OF EXISTING PROJECT
Open Anaconda Powershell Prompt and go to project folder:

    pip install -r requirements.txt --user

## Run main version
Open Anaconda Powershell Prompt and go to project folder:
    
    python main.py --maps Greenland Localhost 'Mapname with spaces' --base_url http://169.51.194.78:31798/

## Run main version with Rendering Mode
Open Anaconda Powershell Prompt and go to project folder:
    
    python main.py --render_mode --maps Greenland Localhost 'Mapname with spaces' --base_url http://169.51.194.78:31798/