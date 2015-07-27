Langevin Model
==============

This is a 1-D Langevin model, originally stemming from ajkluber's repository on github. The model is configured so that it's similar to the design of ajkluber's model builder and project tools repository.

Purpose:
--------

Provie an easily configurable test bed for 1-D Langevin Dynamics for testing new ideas and theories. 


Description:
------------

There are three files of concern:

model.py: Contains the Langevin model class that constructs a simplified model from a .ini file. It's used for storing the potentials, information about what potential functions are guiding the motion. It also stores the params for each potential, so that different potentials can be turned on and off, and fitted.

simulation.py: For organizing and running the simulations and various analysis steps. All methods for that should be in this file.

compute.py: For running langevin dynamics, computing output files and analyzing the results. This is where the heavy computation takes place.



