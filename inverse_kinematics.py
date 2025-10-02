import numpy as np

def arccos_clamped(x):
    """
    Clamp arccos value to [-1, 1]
    """
    return np.arccos(np.clip(x, -1.0, 1.0))

def inverse_kinematics(position) -> list[float]:
    """
    Inverse kinematics of ReactorX-150 Robot Arm.
    """
    # Input: position of end effector [x, y, z]
    # Output: joint angles [joint1, joint2, joint3]

    link1z = 0.065
    link2z = 0.039
    link3x = 0.050
    link3z = 0.150
    link4x = 0.150

    x = position[0]
    y = position[1]
    z = position[2]

    # Solving for joint 1
    joint1 = np.atan2(y, x) # Location of joint 1

    # Solving for joint 2
    r = np.sqrt((x ** 2) + (y ** 2))
    a = np.sqrt((r ** 2) + ((z - link1z - link2z) ** 2))
    gamma = arccos_clamped(r / a)
    k = np.sqrt((link3x ** 2) + (link3z ** 2))
    alpha = arccos_clamped(link3z / k)
    beta_1 = arccos_clamped(((a ** 2) + (k ** 2) - (link4x ** 2)) / (2 * a * k))
    joint2 = (np.pi / 2) - alpha - beta_1 - gamma

    # Solving for joint 3
    eta = np.atan(link3z / link3x)
    beta_2 = arccos_clamped(((k ** 2) + (link4x ** 2) - (a ** 2)) / (2 * k * link4x))
    joint3 = beta_2 - (np.pi - eta)
    
    return [joint1, joint2, joint3]

def main():
    position = [0.2, 0.0, 0.254]
    ik = inverse_kinematics(position)
    print(f"Joint Angles (rad): {ik}")
    print(f"End effector position (x, y, z): {position}")

if __name__ == "__main__":
    main()