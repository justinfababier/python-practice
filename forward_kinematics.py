import numpy as np
import modern_robotics as mr

def forward_kinematics(joints) -> list[float]:
    """
    Forward kinematics of ReactorX-150 Robot Arm.
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

    # Define screw axes
    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([0, 1, 0, -(link1z + link2z), 0, 0])
    S3 = np.array([0, -1, 0, (link1z + link2z + link3z), 0, -link3x])

    # Compute the exponential of the screw coordinates
    S1_skewsym = mr.MatrixExp6(mr.VecTose3(S1 * joint1))
    S2_skewsym = mr.MatrixExp6(mr.VecTose3(S2 * joint2))
    S3_skewsym = mr.MatrixExp6(mr.VecTose3(S3 * joint3))

    M = np.array([[1, 0, 0, (link3x + link4x)],
                  [0, 1, 0, 0],
                  [0, 0, 1, (link1z + link2z + link3z)],
                  [0, 0, 0, 1]])

    # Calculate end effector transformation matrix
    T_04 = S1_skewsym @ S2_skewsym @ S3_skewsym @ M

    # Extract end effector position from the transformation matrix
    x = T_04[0, 3]
    y = T_04[1, 3]
    z = T_04[2, 3]

    # Print transformation matrix
    print(f"T_04 = \n{T_04}")

    return [x, y, z]

def main():
    joints = [0, 0, 0]
    fk = forward_kinematics(joints)
    print(f"Joint angles (rad) = {joints}")
    print(f"End effector position (x, y, z) = {fk}")

if __name__ == "__main__":
    main()