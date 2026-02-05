# Hydropower Equivalent Model for European Countries
This tool is for creating an equivalent hydropower model. The tool is used for the EU project [MOPO](https://www.tools-for-energy-system-modelling.org/). 
With given inflow data and detailed hydropower production or historical production 
## How to use

### Requirement
1. Python 
2. Git
3. Inflow data for a bidding zone (Can be retrieved here for example: (https://github.com/Yil2/hydro_mopo))
4. A detailed hydropower schedule or historical production
5. Prices for the given time span
6. Create a .env with SECRET_FOLDER_PATH= path to the data
7. In this folder, you need three subfolders:
  - Input
  - Equivalent solutions


### Run the tool
For estimation of an area for reservoir-based data, you need to run 01_main_PSO
For estimation of pump equivalent systems, you need to run 02_main_PSO_pump

### User Configuration
All inputs can be changed in the config file that will also be saved at the end of each run 
A template is given in the the configs folder
