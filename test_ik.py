import pytest
import numpy as np
import modern_robotics as mr
from unittest.mock import MagicMock
import sys

sys.path.append(r"/Users/justinfababier/Documents/GitHub/python-practice")
from inverse_kinematics import inverse_kinematics

def test_ik_rest_pos():
    """Test IK with rest coordinates located at (0.2, 0, 0.254)"""
    position = [0.2, 0.0, 0.254]            # Position coordinates (meters)
    result = inverse_kinematics(position)   # Result joint angles (radians)
    expected = [0.0, 0.0, 0.0]              # Expected joint angle values (radians)
    print(result)
    assert np.allclose(result, expected, atol=1e-3)

def test_ik_j1_90deg():
    """Test FK with position at (0, 0.2, 0.254)"""
    position = [0.2, 0.0, 0.254]            # Position coordinates (meters)
    result = inverse_kinematics(position)   # Joint angles (meters)
    expected = [0.0, 0.0, 0.0]              # Expected joint angle values (radians)
    assert np.allclose(result, expected, atol=1e-3)