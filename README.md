# Jack's Car Rental <br/> A Reinforcement Learning Example Using Python
## Introduction
This small Python package solves the Jack's Car Rental problem defined 
in the classic reinforcement book 
**_Reinforcement Learning: An Introduction_** by Sutton and Barto &copy; 2012
(Example 4.2, Chapter 4)

The original problem reads as follows:
> Jack manages two branches of a car rental company. Each day a Poisson-distributed number of customers come to each location and for each car rented out, Jack earns $10. Any car rented is returned at the end of the day but must spend the next day being serviced (not available for renting). Both rentable and serviced cars take up space and each branch has an upper limit to the cars that can be stored. Jack is able to transfer cars between the branches at the cost of $2 per car overnight. Only cars not being serviced can be transferred and transferred cars are available for rent at the new branch the next day.

> **What is the optimal transfer policy of cars between branches?**



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

## Reference:
https://github.com/swiffo/Dynamic-Programming-Car-Rental
Comparing the above solution, the one presented in this repository is much faster because the aforementioned techniques.  

