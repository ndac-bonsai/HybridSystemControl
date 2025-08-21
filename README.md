# HybridSystemControl
Control of hybrid discrete-continuous systems

# Setting up to develop on the Cortical State Inference Package
## Python Environment Setup
1.	Install Python 3.10.11 from the Python website (not the Windows store)
2.	In VSCode, open the folder that you want to develop in
3.	Create virtual python environment & install requirements
````
$ /PATH/TO/PYTHON/python.exe -m venv .venv
$ cd .venv
$ .\\Scripts\activate
$ python -m pip install –upgrade pip setuptools wheel
$ pip install -r requirements.txt
````
4.	Install the ssm library into your virtual environment
````
$ git clone https://github.com/lindermanlab/ssm
$ cd ssm
$ pip install .
````
Visual Studio Project Setup
1.	Open the NDACPython.sln solution in Visual Studio 2022
2.	Link Solution to the python virtual environment you just created
3.	Run project in Bonsai
a.	If you get the error “Unable to start program ‘Bonsai.exe’. The system cannot find the file specified” you can solve this error by moving the example workflows “SimulateLDS.bonsai” and “SimulateControl.bonsai” up one directory and re-run. Put these workflows back in the original directory and run again.
4.	Use “SimulateLDS.bonsai” example workflow to create systems that respond to inputs and test the filtering capabilities of LDS models.  Models are modularly constructed and can be arbitrary discrete/continuous state dimensionality.  Observations can be drawn from any distribution in the ssm library, but easiest to interpret are probably “gaussian” and “poisson”.
a.	The goal of this workflow is to make a “plant” system that generates observations either spontaneously, or in response to input (created with a function generator). Then another system , or “model,” is used to estimate the latent states of the plant given the observations the plant generates and a copy of the inputs, using a Kalman filter for gaussian observations or an Extended-Poisson Kalman filter for poisson observations.  
