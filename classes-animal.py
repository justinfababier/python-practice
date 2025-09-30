from abc import ABC, abstractmethod

class Animal(ABC):
    """
    Represents animal with a name, age, and life-status.
    """
    def __init__(self, name: str, age:int, is_alive: bool):
        """
        Initialize an Animal instance.

        Args:
            name (str): Name of the animal.
            age (int): Age in years.
            is_alive (bool): Life status.
        """
        self.name = name
        self.age = age
        self.is_alive = is_alive

    @abstractmethod
    def eat(self, food_item: str):
        """
        Eat a food item.
        
        Args:
            food_item (str): Name of food item.
        """
        pass

    @abstractmethod
    def sleep(self, time_slept: int):
        """
        Sleep.

        Args:
            time_slept (int): Amount of time slept, in hours.
        """
        pass

    @abstractmethod
    def show_info(self):
        print(f"Name: {self.name}, Age: {self.age}, Alive: {self.is_alive}")

class Human(Animal):
    """
    Represents a human person.
    """
    def __init__(self, name: str, age: int, is_alive: bool, height: int, weight: int, occupation: str):
        """
        Initialize a Human instance.

        Args:
            name (str): Name of the animal.
            age (int): Age in years.
            is_alive (bool): Life status.
            height (int): Height, in inches.
            weight (int): Weight, in lbs.
            Occupation (str): Occupation e.g., student, doctor, unemployed, etc.
        """
        super().__init__(name, age, is_alive)
        self.height = height
        self.weight = weight
        self.occupation = occupation

    def eat(self, food_item: str):
        """
        Eat a food item (VIRTUAL FUNCTION).
        """
        if food_item == None:
            print("No food item in hand!")
            return
        print(f"{self.name} eats {food_item}.")

    def sleep(self, time_slept: int):
        """
        Sleep (VIRTUAL FUNCTION).
        """
        if time_slept == 0 or time_slept == None:
            print(f"{self.name} is awake and has not slept.")
            return
        print(f"{self.name} goes to sleep for {time_slept} hours.")

    def greet(self):
        """
        Person makes a greeting.
        """
        print(f"{self.name} says hello.")

    def show_info(self):
        """
        Show Human's info (VIRTUAL FUNCTION).
        """
        print(f"Name: {self.name}, Age: {self.age}, Alive: {self.is_alive}, Occupation: {self.occupation}")

    def awake(self):
        """
        Awaken.
        """
        print(f"{self.name} is wide awake.")

if __name__ == "__main__":
    person_1 = Human("Justin", 26, True, 65, 120, "Engineer")
    person_1.show_info()
    person_1.eat("eggs")
    person_1.sleep(3)
    person_1.awake()
    person_1.greet()