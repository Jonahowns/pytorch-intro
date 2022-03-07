# pytorch-intro
Small collection of pytorch Models for aptamer sequence data

# What's Included
2 Pytorch Lightning Scripts for training a Resnet and a Variational Auto Encoder

links.txt -> Contains links to Pytorch and other python package Documentation/Tutorials

MLmodels contains some universal functions used by both models, primarily it parses the data for each model

If a model can't find the data, make sure the abs_path variable in get_rawdata() in MLmodels.py has a correct path

# Prep
Before running the scripts make sure you have an anaconda env with all the dependencies
You can make one using the provided exmachina3.yml file
To do use the following command: conda env create --name myenv --file=exmachina3.yml

# Try to the Scripts Locally!
Try to run the scripts!
Both scripts can be called using:

conda activate myenv  # Activate env we created above, you can name it anything 

python resnet.py   # Call Resnet Training 

python VAE.py    # Call VAE Training

## Both scripts are configured to run for 10 epochs
## The only datatype variables currently supported are HCLT, HCRT, and HCB20T

# Try to run the scripts on Agave!
Before launching a job on Agave, we need to configure our python env again
Launch an interactive job with a gpu so our env is setup correctly

on Agave call:
interactive -p sulcgpu1 -q wildfire --gres=gpu:1 -c 6      # Launch interactive job

then load anaconda:
module load anaconda3/4.4.0 

next make our env:
conda env create --name exmachina3 --file=exmachina3.yml    # if you change the name, be sure to update the submission files in agave_submit

Once that's done, let's test the model:
python resnet.py

If it runs, we can exit the interactive job using the 'exit' command

Now let's run it using slurm, make sure to modify the submission files in agave_submit!!
sbatch ./agave_submit/resnet_submit.sh

To check the progess use
squeue -u myusername


