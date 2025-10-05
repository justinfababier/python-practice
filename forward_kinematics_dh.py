import numpy as np
import modern_robotics as mr

def dh_transform(a, alpha, d, phi) -> list[float]:
    """Compute the individual DH transformation matrix."""
    arr = np.array([[np.cos(phi), -np.sin(phi), 0, a],
                    [np.sin(phi) * np.cos(alpha), np.cos(phi) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
                    [np.sin(phi) * np.sin(alpha), np.cos(phi) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
                    [0, 0, 0, 1]])

    return arr

def forward_kinematics(joints) -> list[float]:
    """
    Forward kinematics of ReactorX-150 Robot Arm.
    (This code utilizes the DH Paramters method.)
    """
    # Input: joint angles [joint1, joint2, joint3]
    # Output: the position of end effector [x, y, z]

    # Define link lengths
    link1z = 0.065
    link2z = 0.039
    link3x = 0.050
    link3z = 0.150
    link4x = 0.150
    
    # Define joint angles
    joint1 = joints[0]
    joint2 = joints[1]
    joint3 = joints[2]

    # Define DH table
    DH = [[0, 0, link1z, joint1],
          [0, (-np.pi / 2), -link2z, joint2],
          [link3x, (np.pi / 2), link3z, joint3],
          [link4x, 0, 0, 0]]

    # Calculate link frame transformations
    T_01 = dh_transform(*DH[0])
    T_12 = dh_transform(*DH[1])
    T_23 = dh_transform(*DH[2])
    T_34 = dh_transform(*DH[3])

    # Calculate end effector transformation matrix
    T_04 = T_01 @ T_12 @ T_23 @ T_34

    # Extract end effector position from transformation matrix
    x = T_04[0, 3]
    y = T_04[1, 3]
    z = T_04[2, 3]

    # Print transformation matrix
    print(T_04)

    return[x, y, z]

def main():
    joints = [0, 0, 0]
    fk = forward_kinematics(joints)
    print(f"Joint angles (rad) = {joints}")
    print(f"End effector position (x, y, z) = {fk}")

if __name__ == "__main__":
    main()