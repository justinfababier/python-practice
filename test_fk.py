import pytest
import numpy as np
import modern_robotics as mr
from unittest.mock import MagicMock
import sys

sys.path.append(r"/Users/justinfababier/Documents/GitHub/python-practice")
from forward_kinematics import forward_kinematics

def test_fk_zero_pos():
    """Test FK with all joints at 0 radians."""
    joints = [0, 0, 0]                  # Joint angles (radians)
    result = forward_kinematics(joints) # Result position coordinates (meters)
    expected = [0.2, 0, 0.254]          # Expected joint angle values (meters)
    assert np.allclose(result, expected, atol=1e-6)

def test_fk_j1_90deg():
    """Test FK with joint 1 at pi/2 rad, all other joints at 0 rad."""
    joints = [np.pi / 2, 0, 0]          # Joint angles (radians)
    result = forward_kinematics(joints) # Result position coordinates (meters)
    expected = [0, 0.2, 0.254]          # Expected joint angle values (meters)
    assert np.allclose(result, expected, atol=1e-6)