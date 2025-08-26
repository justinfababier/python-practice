from abc import ABC, abstractmethod

class Engine:
    """
    Represents a car engine with a type and horsepower.

    Attributes:
        _type (str): Type of engine (e.g., Gasoline, Electric).
        _horsepower (int): Engine horsepower.
    """

    def __init__(self, type_: str, horsepower: int):
        """
        Initialize an Engine instance.

        Args:
            type_ (str): The type of the engine (Gasoline or Electric).
            horsepower (int): The horsepower of the engine.
        """
        self._type = type_
        self._horsepower = horsepower

    def start(self):
        """Start the engine and print a message."""
        print(f"Starting {self._type} engine...")

    def stop(self):
        """Stop the engine and print a message."""
        print(f"Stopping {self._type} engine...")

    @property
    def horsepower(self):
        """Return the horsepower of the engine."""
        return self._horsepower

class Car(ABC):
    """
    Abstract base class for cars.

    Attributes:
        _brand (str): The car's manufacturer brand.
        _model (str): The car's model name.
        _year (int): The car's production year.
        _mileage (int): The car's odometer reading in miles.
    """

    def __init__(self, brand: str, model: str, year: int, mileage: int):
        """
        Initialize a Car instance.

        Args:
            brand (str): The brand of the car.
            model (str): The model of the car.
            year (int): The year of manufacture.
            mileage (int): The odometer reading in miles.
        """
        self._brand = brand
        self._model = model
        self._year = year
        self._mileage = mileage

    @abstractmethod
    def drive(self, driven_miles: int):
        """
        Drive the car a certain number of miles.

        Args:
            driven_miles (int): The distance to drive in miles.
        """
        pass

    @abstractmethod
    def refuel_or_recharge(self):
        """Refuel (for gas cars) or recharge (for electric cars)."""
        pass

    @abstractmethod
    def remaining_fuels_battery(self):
        """Display the remaining fuel or battery level."""
        pass

    def show_info(self):
        """Print the car's basic information and odometer reading."""
        print(f"{self._year} {self._brand} {self._model}")
        print(f"Odometer: {self._mileage} miles")

class ElectricCar(Car):
    """
    Represents an electric car.

    Attributes:
        _battery_capacity (int): The current battery capacity (max = 100).
        _engine (Engine): The electric engine instance.
    """

    def __init__(self, brand: str, model: str, year: int, mileage: int, battery_capacity: int):
        """
        Initialize an ElectricCar instance.

        Args:
            brand (str): The brand of the car.
            model (str): The model of the car.
            year (int): The year of manufacture.
            mileage (int): The odometer reading in miles.
            battery_capacity (int): The current battery capacity (0–100 kWh).
        """
        super().__init__(brand, model, year, mileage)
        self._battery_capacity = battery_capacity
        self._engine = Engine("Electric", 300)

    def drive(self, driven_miles: int):
        """
        Drive the car for a certain distance, draining battery accordingly.

        Args:
            driven_miles (int): The distance to drive in miles.
        """
        if self._battery_capacity <= 0:
            print("Battery empty! Please recharge.")
            return
        self._engine.start()
        print("Driving with electric engine...")
        self._mileage += driven_miles
        self._battery_capacity -= driven_miles * 0.25
        if self._battery_capacity < 0:
            self._battery_capacity = 0
        print(f"Distance achieved: {driven_miles} miles")
        self._engine.stop()
        self.show_info()
        self.remaining_fuels_battery()

    def refuel_or_recharge(self):
        """Recharge the car's battery to full capacity."""
        print("Recharging battery...")
        self._battery_capacity = 100
        self.remaining_fuels_battery()

    def remaining_fuels_battery(self):
        """Print the remaining battery level."""
        print(f"Remaining battery: {self._battery_capacity:.2f} kWh")

class GasCar(Car):
    """
    Represents a gasoline-powered car.

    Attributes:
        _fuel_tank_size (int): Maximum fuel tank capacity in gallons.
        _fuel (float): Current fuel level in gallons.
        _engine (Engine): The gasoline engine instance.
    """

    def __init__(self, brand: str, model: str, year: int, fuel_tank_size: int, fuel: int, mileage: int):
        """
        Initialize a GasCar instance.

        Args:
            brand (str): The brand of the car.
            model (str): The model of the car.
            year (int): The year of manufacture.
            fuel_tank_size (int): The maximum fuel tank capacity in gallons.
            fuel (int): Current fuel level in gallons.
            mileage (int): The odometer reading in miles.
        """
        super().__init__(brand, model, year, mileage)
        self._fuel_tank_size = fuel_tank_size
        self._fuel = min(fuel, fuel_tank_size)
        self._engine = Engine("Gasoline", 200)

    def drive(self, driven_miles: int):
        """
        Drive the car for a certain distance, consuming fuel.

        Args:
            driven_miles (int): The distance to drive in miles.
        """
        fuel_needed = driven_miles * 0.25
        if self._fuel < fuel_needed:
            print("Not enough fuel! Please refuel.")
            return
        self._engine.start()
        print("Driving with gas engine...")
        self._mileage += driven_miles
        self._fuel -= fuel_needed
        print(f"Distance achieved: {driven_miles} miles")
        self._engine.stop()
        self.show_info()
        self.remaining_fuels_battery()

    def refuel_or_recharge(self):
        """Refill the fuel tank to its maximum capacity."""
        print("Refueling gas tank...")
        self._fuel = self._fuel_tank_size
        self.remaining_fuels_battery()

    def remaining_fuels_battery(self):
        """Print the remaining fuel level."""
        print(f"Remaining fuel: {self._fuel:.2f} gallons")

if __name__ == "__main__":
    car1 = GasCar("Toyota", "Camry", 2022, 16, 10, 20000)
    car1.drive(20)
    car1.refuel_or_recharge()

    print("\n---\n")

    car2 = ElectricCar("Tesla", "Model 3", 2022, 5000, 100)
    car2.drive(20)
    car2.refuel_or_recharge()
