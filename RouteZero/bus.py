"""
         Module for defining a bus class to hold bus parameters
"""


class Bus:
   def __init__(self, max_passengers, battery_capacity, charging_rate, gross_mass,
              charging_efficiency=0.95, end_of_life_cap=0.8):
      """
      initialises a bus object
      :param max_passengers: max number of passengers
      :param battery_capacity: battery capcity (kWH)
      :param charging_rate: maximum charging rate (kW)
      :param gross_mass: gross mass of the bus i.e. fully loaded (kg)
      :param charging_efficiency: (default=0.95) charging efficiency [0->1]
      :param end_of_life_cap: (default=0.8) percentage of battery capacity still remaining at end of battery life [0->1]
      """
      self.max_passengers = max_passengers
      self.battery_capacity = battery_capacity
      self.charging_rate = charging_rate
      self.gross_mass = gross_mass
      self.net_mass = gross_mass - 65 * max_passengers
      self.charging_efficiency = charging_efficiency
      self.end_of_life_cap = end_of_life_cap
      self.usable_capacity = battery_capacity * end_of_life_cap
      self.soc = 1. * self.usable_capacity

   def get_soc_percent(self):
      " Returns current state of charge as a percentage"
      return self.soc / self.usable_capacity * 100

class Yutong(Bus):
   """
   Paraemters for the Yutong bus used in the Leichhardt trial
   """
   def __init__(self):
      Bus.__init__(self,74, 422, 300, 18000)

class BYD(Bus):
   """
   Parameters for the BYD bus
   """
   def __init__(self):
      Bus.__init__(self,74, 368, 80, 18000)

if __name__=="__main__":
   bus = BYD()