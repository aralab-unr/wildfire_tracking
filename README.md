# wildfire_tracking

To run the DQN with state estimators:
```
python3 state_est_dqn_obsv_test.py
```

The fire simulation files are written in fbndry4.txt and fbndry5.txt files. 
They are represented as text files and imported from our farsite simulator so the simulator does not have to step through the fire simulation. 
This is a very exhaustive process and computationally expensive process and should be done before hand as exported to text files 
for ease of use and to make the learning and simulation process fast.
