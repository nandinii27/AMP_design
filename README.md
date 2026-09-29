
# Joint Optimization of a Battery System and Wind Farm

## Context

The integration of renewable energy sources into power systems, such as wind energy, presents both challenges and opportunities. The main challenge is the variability of production, which may not coincide with periods of highest demand or electricity market prices. To better manage this variability and maximize revenues, it is common to complement wind farms with energy storage systems, such as batteries. These systems allow energy to be stored when wind production is high and prices are low, and released when prices are higher.

## Problem Description

The objective is to maximize the joint revenue of a wind farm and a battery over a one-day horizon (24 hours). The optimal operating strategy for the battery (when to charge and discharge) must be determined, considering both wind production and hourly electricity market prices.

The battery operation strategy is determined the day before, based on the forecasted wind production. Since production varies and cannot be predicted exactly, 13 forecast scenarios for the wind farm are provided (all equally probable). The optimal strategy should not be based on a single scenario, but should maximize the expected revenue across all scenarios.

The battery can be charged from both the wind farm generation and the grid. Both facilities are considered as a portfolio, combining their revenues and costs to obtain a joint margin.

## Available Data

**File:** `input_data.csv`

File structure:

- **hour:** Hour of the day (1 to 24).
- **scenario_1 to scenario_13:** Wind production forecasts (MWh) for the 13 equally probable scenarios.
- **price:** Electricity market price (€/MWh), the same for all scenarios.

This format allows modeling the uncertainty in wind production and facilitates optimal decision-making considering both the variability of renewable generation and the evolution of electricity prices.

## Battery Physical Characteristics

- Maximum storage capacity: **16 MWh**
- Maximum charging power: **5.0 MW**
- Maximum discharging power: **4.0 MW**
- Charging efficiency: **80%** (for every 5 MW consumed from the grid during charging, only 4 MW are actually stored)
- Discharging efficiency: **100%**
- Maximum number of full charge-discharge cycles per day: **2**
- The battery starts discharged and must end the day discharged.
- No additional costs are considered for purchasing energy from the grid.

## Exercise Requirements

- Formulate the optimization model to decide, hour by hour, how much energy the battery should store and how much should be discharged for sale.
- Consider the uncertainty in wind production using the 13 equally probable scenarios. The objective is to maximize the expected revenue.
- Include all the physical constraints of the battery and the initial and final state condition (battery discharged at the beginning and end of the day).
- Each hour, the battery can only be charging or discharging, not both at the same time.
- For simplicity, the battery is considered to buy and sell all its energy from the grid at the market price, without netting with the wind farm.
- Present and justify modeling decisions (variables, objective function, constraints, etc.).
- Solve the problem with the tool you consider appropriate and interpret the results. Document and justify any simplification made.

## Deliverables

- Explanatory document of the model, variables, and constraints.
- Code used to solve the problem.
- Results obtained and their interpretation.
- Optional: sensitivity analysis regarding changes in battery parameters or market prices.

## Additional Notes

- It is assumed that the battery and the wind farm are connected at the same grid connection point and can operate in a coordinated manner.
- No additional energy losses are considered beyond the specified charging efficiency.
- Justify any additional assumptions necessary for solving the problem.

---

Good luck! If you have any questions about the data or the approach, do not hesitate to ask.
