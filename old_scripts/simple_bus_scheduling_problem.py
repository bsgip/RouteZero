##
import numpy as np

# a 5 trip scheduling example
num_trips = 5

# number of trips including null trip
N = num_trips+1

# define some required trip start times (null trip included as nan)?
ts = np.array([np.nan,0, 5, 10, 9, 14])

# define trip finish times
tf = np.array([np.nan, 3, 8, 13, 12, 17])

# define connecting travel times (this needs to include to and from the null route
tc = 1.5*np.ones((N,N))
tc[:,0] = 3.
tc[0, :] = 3.

import pyomo.environ as pyEnv

#Model
model = pyEnv.ConcreteModel()

# indices for the trips including the null trip
model.N = pyEnv.RangeSet(N)

#Index for the dummy variable u
model.U = pyEnv.RangeSet(2,N)

#Decision variables xij
model.x = pyEnv.Var(model.N,model.N, within=pyEnv.Binary)

#Dummy variable ui
model.u = pyEnv.Var(model.N, within=pyEnv.NonNegativeIntegers,bounds=(0,N-1))

#Cost Matrix cij
model.tc = pyEnv.Param(model.N, model.N,initialize=lambda model, i, j: tc[i-1][j-1])

# objective function
def obj_func(model):
    return sum(sum(model.x[i,j] * model.tc[i,j] for i in model.N) for j in model.N)

model.objective = pyEnv.Objective(rule=obj_func,sense=pyEnv.minimize)

def rule_const1(model,N):
    if N==1:
        return pyEnv.Constraint.Skip
    else:
        return sum(model.x[i,N] for i in model.N if i!=N ) == 1

model.const1 = pyEnv.Constraint(model.N,rule=rule_const1)

def rule_const2(model,N):
    if N==1:
        return pyEnv.Constraint.Skip
    else:
        return sum(model.x[N,j] for j in model.N if j!=N) == 1

model.rest2 = pyEnv.Constraint(model.N,rule=rule_const2)


def rule_const3(model, i, j):
    if i != j:
        return model.u[i] - model.u[j] + model.x[i, j] * N <= N - 1
    else:
        # Yeah, this else doesn't say anything
        return model.u[i] - model.u[i] == 0

model.rest3 = pyEnv.Constraint(model.U, model.N, rule=rule_const3)

# now include the cant start trip before we have travelled from previous trip constraint
def rule_const4(model, i, j):
    if (i == 1) or (j == 1):
        return pyEnv.Constraint.Skip
    else:       # the minus ones are to convert to zero based indexing for numpy stuff
        return ts[j-1] - tf[i-1] - tc[i-1,j-1] + (1 - model.x[i,j]) * 100. >= 0.

model.rest = pyEnv.Constraint(model.N, model.N, rule=rule_const4)

#Solves
solver = pyEnv.SolverFactory('cplex')
result = solver.solve(model,tee = False)

#Prints the results
print(result)

l = list(model.x.keys())
for i in l:
    if model.x[i]() != 0:
        print(i,'--', model.x[i]())