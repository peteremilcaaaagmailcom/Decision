from pyomo.environ import *
import numpy as np
import pandas as pd
from data import get_fixed_data as get
from WindProcess import wind_model as wind
from PriceProcess import price_model as price
import data
import matplotlib.pyplot as plt

# Hent faste parametre
data = get()

# Initial værdier
wind_values = [0.1]  # Startværdi for vind
price_values = [10]  # Startværdi for pris

# Simuler 24 tidsperioder
for t in range(1, 24):
    next_wind = wind(wind_values[-1], wind_values[-2] if t > 1 else wind_values[-1], data)
    next_price = price(price_values[-1], price_values[-2] if t > 1 else price_values[-1], next_wind, data)

    wind_values.append(next_wind)
    price_values.append(next_price)

# Define model
model = ConcreteModel()

# Define sets
T = range(0,data['num_timeslots']) # Time periods (e.g., 24 hours)
model.T = Set(initialize=T)

# Define parameters
lambda_grid = price_values  # Cost per unit power from grid
Celzr = data['electrolyzer_cost']  # Fixed cost for electrolyzer use
P2H = data['p2h_rate'] # Max power to hydrogen conversion
H2P = data['h2p_rate'] # Max power from fuel cell
C = data['hydrogen_capacity']    # Max storage capacity
Rp2h = data['conversion_p2h'] # Efficiency of power-to-hydrogen
Rh2p = data['conversion_h2p'] # Efficiency of hydrogen-to-power
M = max(wind_values)   # Large constant for binary constraint
D = data['demand_schedule']  # Power demand at time t

# Define decision variables
model.xgrid = Var(T, within=NonNegativeReals)
model.xwt = Var(T, within=NonNegativeReals)
model.xh2p = Var(T, within=NonNegativeReals)
model.xp2h = Var(T, within=NonNegativeReals, bounds=(0, P2H))
model.xcap = Var(T, within=NonNegativeReals, bounds=(0, C))
model.xelzr = Var(T, within=Binary)

# Objective function: Minimize total cost
model.obj = Objective(
    expr=sum(lambda_grid[t] * model.xgrid[t] + model.xelzr[t] * Celzr for t in T),
    sense=minimize
)

# Constraints
# Power balance constraint
model.power_balance = ConstraintList()
for t in T:
    model.power_balance.add(model.xgrid[t] + model.xwt[t] + model.xh2p[t] == D[t])

#Wind power constraint
model.Wind_power = ConstraintList()
for t in T:
    model.Wind_power.add(model.xwt[t] + model.xp2h[t] <= wind_values[t])

# Constraints on power conversion
model.hydrogen_conversion = ConstraintList()
for t in T:
    model.hydrogen_conversion.add(model.xh2p[t] <= H2P)
    model.hydrogen_conversion.add(model.xp2h[t] <= P2H)
    model.hydrogen_conversion.add(model.xp2h[t] <= M * model.xelzr[t])

# Storage constraint
model.storage = ConstraintList()
for t in T:
    if t > 1:
        model.storage.add(
            model.xcap[t] == model.xp2h[t] * Rp2h - model.xh2p[t] * Rh2p + model.xcap[t-1]
        )
    else:
        model.storage.add(model.xcap[t] == model.xp2h[t] * Rp2h - model.xh2p[t] * Rh2p)

# Create a solver
solver = SolverFactory('gurobi')  # Make sure Gurobi is installed and properly configured

# Solve the model
results = solver.solve(model, tee=True)

# Check if an optimal solution was found
if results.solver.termination_condition == TerminationCondition.optimal:
    print("Optimal solution found")

    # Print out variable values and objective value
    print("Variable values:")
    for t in T:
        print(f"Time {t}: xgrid={value(model.xgrid[t]):.3f}, xwt={value(model.xwt[t]):.3f}, xp2h={value(model.xp2h[t]):.3f},  xh2p={value(model.xh2p[t]):.3f}, xcap={value(model.xcap[t]):.3f}, xelzr={int(value(model.xelzr[t]))}")
    print(f"\nObjective value: {value(model.obj):.3f}\n")
else:
    print("No optimal solution found.")

print("Wind Values:", wind_values)
print("Price Values:", price_values)

initial_state = {'hydrogen': 0, 'electrolyzer_status': 0}

times = range(data['num_timeslots'])

# Plot results
plt.figure(figsize=(14, 10))

plt.subplot(9, 1, 1)
plt.plot(times, wind_values, label="Wind Power", color="blue")
plt.ylabel("Wind Power")
plt.legend()

plt.subplot(9, 1, 2)
plt.plot(times, data['demand_schedule'], label="Demand Schedule", color="orange")
plt.ylabel("Demand")
plt.legend()

xelzr_values = [value(model.xelzr[t]) for t in times]
plt.subplot(9, 1, 3)
plt.step(times, xelzr_values, label="Electrolyzer Status", color="red", where="post")
plt.ylabel("El. Status")
plt.legend()

xcap_values = [value(model.xcap[t]) for t in times]
plt.subplot(9, 1, 4)
plt.plot(times, xcap_values, label="Hydrogen Level", color="green")
plt.ylabel("Hydr. Level")
plt.legend()

xp2h_values = [value(model.xp2h[t]) for t in times]
plt.subplot(9, 1, 5)
plt.plot(times, xp2h_values, label="p2h", color="orange")
plt.ylabel("p2h")
plt.legend()

xh2p_values = [value(model.xh2p[t]) for t in times]
plt.subplot(9, 1, 6)
plt.plot(times, xh2p_values, label="h2p", color="blue")
plt.ylabel("h2p")
plt.legend()

xgrid_values = [value(model.xgrid[t]) for t in times]
plt.subplot(9, 1, 7)
plt.plot(times, xgrid_values, label="Grid Power", color="green")
plt.ylabel("Grid Power")
plt.legend()

xwt_values = [value(model.xwt[t]) for t in times]
plt.subplot(9, 1, 8)
plt.plot(times, xwt_values, label="wt Power", color="orange")
plt.ylabel("wt power")
plt.legend()

plt.subplot(9, 1, 9)
plt.plot(times, price_values, label="price", color="red")
plt.ylabel("Price")
plt.xlabel("Time")
plt.legend()

plt.tight_layout()
plt.show()
