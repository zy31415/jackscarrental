# Jack's Car Rental <br/> A Reinforcement Learning Example Using Python
## Introduction
This small Python package solves the Jack's Car Rental problem defined 
in the classic reinforcement book 
**_Reinforcement Learning: An Introduction_** by Sutton and Barto &copy; 2012
(Example 4.2, Chapter 4)

The original problem reads as follows:
> **Example 4.2**: 
Jack’s Car Rental Jack manages two locations for a nationwide car rental company.
Each day, some number of customers arrive at each location to rent cars. 
If Jack has a car available, he rents it out and is
credited $10 by the national company. 
If he is out of cars at that location, then the business is lost.
Cars become available for renting the day after they are returned.
To help ensure that cars are available where they are needed,
Jack can move them between the two locations overnight, at a cost of $2 per car moved. 
We assume that the number of cars requested and returned at each location are Poisson random variables, 
meaning that the probability that the number is n is λ^n * e^(−λ) / n!, 
where λ is the expected number. 
Suppose λ is 3 and 4 for rental requests at the first and second locations and 3 and 2 for returns. 
To simplify the problem slightly, we assume that there can be no more than 20 cars at each location 
(any additional cars are returned to the nationwide company, 
and thus disappear from the problem) 
and a maximum of five cars can be moved from one location to the other in one night. 
We take the discount rate to be γ = 0.9 and formulate this as a continuing finite MDP, 
where the time steps are days, 
the state is the number of cars at each location at the end of the day, 
and the actions are the net numbers of cars moved between the two locations overnight.

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

### Performance
Two techniques are used to improve performance:
- Results from scipy.stats.poisson are cached.
- Use numpy matrix operation wherever possible to avoid python loop, which is slow.

### Problem interpretation: When requests greater than availability
When the requests are greater than available cars, in Sutton's original problem, I believe that the reward is zero.
However, in my solution here, I interpret the situation differently:
I give the rewards as the rewards that allowed by the available cars.
This complicates the computation.

### Bad action punishment: when try to move more cars than there is
Add punishment for actions that try to move more cars than there is in one location, 
otherwise the best policy always try to move maximum allowed cars when there is not enough cars available in one place. 

## Reference:
- https://github.com/swiffo/Dynamic-Programming-Car-Rental

Comparing to the above solution, the one presented in this repository is much faster because the aforementioned techniques.  

