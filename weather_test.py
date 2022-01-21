# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly, Stations

# Set time period
start = datetime(2021, 1, 1)
# end = datetime(2018, 1, 30)
end = datetime(2021, 12, 31)

# Create Point for Vancouver, BC  (-32.8960021, 151.7349089)
# location = Point(49.2497, -123.1193, alt=70)
# create point for mayfield east
location = Point(-32.8960021, 151.734908, alt=10)
location.method='weighted'   # method can also be nearest (default) or weighted

# Get daily data for 2018
# data = Daily(location, start, end)
# data = data.fetch()
data = Daily(location,start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
# data.plot(y=['temp'])
data.plot(y=['tavg', 'tmin', 'tmax'])
plt.show()