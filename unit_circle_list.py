import numpy as np

def main():
    """
    Simple program that:
        (1) Defines a Python list of unit circle values
        (2) Traverses through the list to:
            - Calculate cartesian coordinates, given a predefined radius value
            - Value of degrees in radians
            - Print formatted values to the console as an f-string
    """
    my_list = [0, 90, 180, 270, 360]    # Contains values in degrees

    for i, deg in enumerate(my_list):
        rad = deg * np.pi / 180 # Value of degrees in radians
        r = 1   # Radius length
        x = r * np.cos(deg) # x-coordinate
        y = r * np.sin(deg) # y-coordinate

        # Print to console
        print(f"Index: {i} | Degrees: {deg:.1f} | Radians: {rad:.3f} | Coordinates: ({x:.3f}, {y:.3f})")

if __name__ == "__main__":
    main()