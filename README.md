# Recursive_Least-Squares_Algorithm
This repository is an attempt to solve recursive least square algorithm problem to design an adaptive filter.


Availble data [x,y]
Description:
1- One main system takes in x gives out y
 X --> ||       Main System     ||   --> Y

2- Other is RLS System 

RLS will take the inputs of x and prerecorded y
By using arbitary definitions of a polynomial system RLS will find the 
best suitble cooefficents of that arbitary system (which was predetermined)

f(X) = [ [X[M]^N,X[M]^N-1,X[M]^N-2,X[M]^N-3,...] ,[[X[M-1]^N,X[M-1]^N-1,X[M-1]^N-2,X[M-1]^N-3,...]],.. ] 
<Regressors>

 [f(X),Y] --> ||        RLS            ||   -->  W

It is designers job to predict best description of parameters for the RLS
Memory(M) and Degree(N) or (M*N == Number of parameters), Inital Conditions, Regressor Choice 

It is not as intuational at so it is fine to repaet basic concept of this system:

The goal is to mathematically model a non linear system 
Reference data is physically measured data which we call y.
Those reference data is generated by specific input which is essential to model the system thus it is also stored in somewhere.
Weigths are genereated according to those x and y values.
Noteworthy fact about the computation, 
While generating a random input just to test the algorithm 
What you create essentially a linear equation + nonlinear equation(random noise)
If the nonlinear contribution is much higher than the linear equation then the modelling would not be as accurate
as it normally should. The reason behind this is final estimation of the system is a **linear** equation if the nonlinear elements have large effect rls would constintly try to fit that randomness into a linear equation. As it has been mentioned before noise is not something can be expressed as linear equation. In fact best description of a noise can be gaussian distribution. Keep in mind that there migth be different version of noise such as noise which repates it self, noise that have relationship with the input etc. the nature of it may give a justification to disgard some of my commentary. My commentary is based upon different trails of the algorithm with different kinds of noises, feel free to test it yourselves.




