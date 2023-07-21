This is a brief readme for those want to use the scripts here


1. the conda environment package is environment.yml, which should contains all packages required to run the script


2. the main file is _main.py (i.e. transfer_main.py), which will run the experiment and evaluate the network's performance. Most of these files are using deterministic behaviors to ensure reproducibility, so you will need to add 'CUBLAS_WORKSPACE_CONFIG=:4096:8' before the 'python _main.py' command. (i.e. 'CUBLAS_WORKSPACE_CONFIG=:4096:8 python transfer_main.py'


3. 2 locations are needed for specifying the data: the location (specified in cnn_config.json), and the metadata that contains labels information (specified in dataloader.py). Therefore, if you are using your own dataset, you should let the dataloader know how to read them correctly - by make corresponding modifications to the dataloader.py. Similarly, if you are using different way of metadata files, you should modify the way to read in those data.


4. GPU is required by default, however, if you want to run on cpu, please make corresponding modifications in networks.py


5. make sure checkpoint_dir exists before running the main program for now. In future updates, this restriction may be lifted by automatic detection.
