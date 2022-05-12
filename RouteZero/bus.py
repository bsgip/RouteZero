


class Bus:
   def __init__(self, max_passengers, battery_capacity, charging_rate, gross_mass,
              charging_efficiency=0.9, end_of_life_cap=0.8):
      """
      initialises a bus object
      :param max_passengers: max number of passengers
      :param battery_capacity: battery capcity (kWH)
      :param charging_rate: maximum charging rate (kW)
      :param gross_mass: gross mass of the bus i.e. fully loaded (kg)
      :param charging_efficiency: (default=0.9) charging efficiency [0->1]
      :param end_of_life_cap: (default=0.9) percentage of battery capacity still remaining at end of battery life [0->1]
      """
      self.max_passengers = max_passengers
      self.battery_capacity = battery_capacity
      self.charging_rate = charging_rate
      self.gross_mass = gross_mass
      self.net_mass = gross_mass - 65 * max_passengers
      self.charging_efficiency = charging_efficiency
      self.end_of_life_cap = end_of_life_cap
      self.usuable_capacity = battery_capacity * end_of_life_cap
      self.soc = 1*self.usuable_capacity

class Yutong(Bus):
   def __init__(self):
      Bus.__init__(self,74, 422, 300, 18000)

class BYD(Bus):
   def __init__(self):
      Bus.__init__(self,74, 368, 80, 18000)


if __name__=="__main__":
   bus = BYD()