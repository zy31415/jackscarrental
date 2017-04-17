# Jack's Car Rental <br/> A Reinforcement Learning Example Using Python
## Introduction
This small Python package solves the Jack's Car Rental problem defined 
in the classic reinforcement book 
**_Reinforcement Learning: An Introduction_** by Sutton and Barto &copy; 2012
(Example 4.2, Chapter 4)

## Run the code
- Language: python3
- Required libraries: 
    - *numpy*, *scipy* for numerical and statistical computing; 
    - *matplotlib* for plotting.

I suggest you use virtual environment to run the code:
```bash
pyvenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Make sure that the library can be found by python
```bash
export PYTHONPATH = /path/to/jackscarrental/
```

Now you are ready to run the demo.
Go to the tests directory and type:
```bash
python scripts_you_want_to_run.py
```
## Code structure
All core computational codes are under jackscarrental/ directory 
as a python library.

Test, example and demo codes are under tests/ directory.

## Implementation details

Two techniques are used to improve performance.
- Results from scipy.stats.poisson are cached.
- Use numpy matrix operation wherever possible to avoid python loop, which is slow.