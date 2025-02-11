import numpy as np
import pandas as pd
from data import get_fixed_data as get
from WindProcess import wind_model as wind
from PriceProcess import price_model as price

# Hent faste parametre
data = get()

# Initial værdier
wind_values = [5]  # Startværdi for vind
price_values = [35]  # Startværdi for pris

# Simuler 24 tidsperioder
for t in range(1, 24):
    next_wind = wind(wind_values[-1], wind_values[-2] if t > 1 else wind_values[-1], data)
    next_price = price(price_values[-1], price_values[-2] if t > 1 else price_values[-1], next_wind, data)

    wind_values.append(next_wind)
    price_values.append(next_price)

# Gem som CSV
df = pd.DataFrame({"Timestep": np.arange(24), "Wind": wind_values, "Price": price_values})

