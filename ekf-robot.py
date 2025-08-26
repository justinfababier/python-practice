import numpy as np
import matplotlib.pyplot as plt

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) implementation.
    """

    def __init__(self, x0, P0, Q, R, f, h, F_jac = None, H_jac = None, eps_jac = 1e-5):
        """
        Initialize the Extended Kalman Filter (EKF).

        Parameters
        ----------
        x0 : array_like
            Initial state vector (shape: n,).
        P0 : array_like
            Initial state covariance matrix (shape: n x n).
        Q : array_like
            Process noise covariance matrix (shape: n x n).
        R : array_like
            Measurement noise covariance matrix (shape: m x m).
        f : callable
            State transition function: f(x, u, dt) -> x_next.
        h : callable
            Measurement function: h(x) -> z.
        F_jac : callable, optional
            Function to compute the Jacobian of f w.r.t. x (analytic), 
            defaults to None (numerical Jacobian is used if not provided).
        H_jac : callable, optional
            Function to compute the Jacobian of h w.r.t. x (analytic), 
            defaults to None (numerical Jacobian is used if not provided).
        eps_jac : float, optional
            Small perturbation for numerical Jacobian calculation, default is 1e-5.

        Raises
        ------
        AssertionError
            If P0 or Q do not have shape (n x n).
        """
        self.x = x0.astype(float).reshape(-1)
        self.P = P0.astype(float)
        self.Q = Q.astype(float)
        self.R = R.astype(float)
        self.f = f
        self.h = h
        self.F_jac = F_jac
        self.H_jac = H_jac
        self.eps = float(eps_jac)

        self.n = self.x.size
        assert self.P.shape == (self.n, self.n), "P0 must be n x n"
        assert self.Q.shape == (self.n, self.n), "Q must be n x n"

    def _numerical_jacobian_f(self, x, u, dt):
        """
        Compute the numerical Jacobian of the state transition function f
        with respect to the state vector x using finite differences.

        Parameters
        ----------
        x : array_like
            Current state vector (shape: n,).
        u : array_like
            Control input vector applied at this step.
        dt : float
            Time step for the state transition.

        Returns
        -------
        J : ndarray
            Numerical Jacobian matrix of f with respect to x (shape: n x n).

        Notes
        -----
        Uses a small perturbation (self.eps) in each state dimension to 
        approximate partial derivatives: J[:, i] ≈ (f(x + eps_i) - f(x)) / eps.
        """
        fx0 = self.f(x, u, dt)
        J = np.zeros((self.n, self.n))

        for i in range(self.n):
            dx = np.zeros_like(x)
            dx[i] = self.eps
            fx1 = self.f(x + dx, u, dt)
            J[:, i] = (fx1 - fx0) / self.eps
        
        return J

    def _numerical_jacobian_h(self, x):
        """
        Compute the numerical Jacobian of the measurement function h
        with respect to the state vector x using finite differences.

        Parameters
        ----------
        x : array_like
            Current state vector (shape: n,).

        Returns
        -------
        J : ndarray
            Numerical Jacobian matrix of h with respect to x (shape: m x n),
            where m is the dimension of the measurement vector.

        Notes
        -----
        Uses a small perturbation (self.eps) in each state dimension to 
        approximate partial derivatives: J[:, i] ≈ (h(x + eps_i) - h(x)) / eps.
        """
        z0 = self.h(x)
        m = z0.size
        J = np.zeros((m, self.n))

        for i in range(self.n):
            dx = np.zeros_like(x)
            dx[i] = self.eps
            z1 = self.h(x + dx)
            J[:, i] = (z1 - z0) / self.eps
        
        return J

    def predict(self, u = None, dt = 1.0):
        """
        Perform the EKF prediction step.

        Propagates the current state estimate through the nonlinear state
        transition function `f` and updates the state covariance `P` using
        the Jacobian of `f`.

        Parameters
        ----------
        u : array_like, optional
            Control input vector applied at this step. Defaults to None.
        dt : float, optional
            Time step for the prediction. Default is 1.0.

        Notes
        -----
        - Uses the analytical Jacobian `F_jac` if provided, otherwise falls
        back to numerical differentiation.
        - Ensures the state covariance `P` remains symmetric.
        """
        self.x = self.f(self.x, u, dt)

        if self.F_jac is not None:
            F = self.F_jac(self.x, u, dt)

        else:
            F = self._numerical_jacobian_f(self.x, u, dt)
        
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q
        self.P = (self.P + self.P.T) / 2.0  # enforce symmetry

    def update(self, z):
        """
        Perform the EKF update (correction) step with a measurement.

        Incorporates a new measurement `z` to correct the predicted state
        estimate, computing the Kalman gain and updating both the state
        vector `x` and covariance `P`.

        Parameters
        ----------
        z : array_like
            Measurement vector at the current step.

        Notes
        -----
        - Uses the analytical measurement Jacobian `H_jac` if provided, 
        otherwise uses numerical differentiation.
        - Handles singular `S` matrices by using the pseudo-inverse if needed.
        - Maintains symmetry of the updated covariance `P`.
        """
        z = np.asarray(z).reshape(-1)
        z_pred = self.h(self.x)

        if self.H_jac is not None:
            H = self.H_jac(self.x)
        
        else:
            H = self._numerical_jacobian_h(self.x)

        y = z - z_pred
        S = np.dot(np.dot(H, self.P), H.T) + self.R

        try:
            K = np.dot(self.P, H.T).dot(np.linalg.inv(S))
        
        except np.linalg.LinAlgError:
            K = np.dot(np.dot(self.P, H.T), np.linalg.pinv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        KH = np.dot(K, H)
        self.P = np.dot(np.dot(I - KH, self.P), (I - KH).T) + np.dot(np.dot(K, self.R), K.T)
        self.P = (self.P + self.P.T) / 2.0

    def set_process_noise(self, Q):
        """
        Set the process noise covariance matrix.

        Parameters
        ----------
        Q : ndarray
            Process noise covariance matrix (shape: n x n).

        Raises
        ------
        AssertionError
            If `Q` does not have shape (n x n).
        """
        assert Q.shape == (self.n, self.n)
        self.Q = Q

    def set_measurement_noise(self, R):
        """
        Set the measurement noise covariance matrix.

        Parameters
        ----------
        R : ndarray
            Measurement noise covariance matrix (shape: m x m).

        Raises
        ------
        AssertionError
            If `R` is not square.
        """
        assert R.shape[0] == R.shape[1]
        self.R = R

    def state(self):
        """
        Return a copy of the current state estimate.

        Returns
        -------
        x : ndarray
            Current state vector.
        """
        return self.x.copy()

    def covariance(self):
        """
        Return a copy of the current state covariance matrix.

        Returns
        -------
        P : ndarray
            Current state covariance matrix.
        """
        return self.P.copy()

def f_robot(x, u, dt):
    """
    Model function f

    Discrete-time nonlinear motion model (position-first state).
    State x = [px, py, vx, vy, theta]
    Control u = [ax_body, ay_body, omega]
    """

    px, py, vx, vy, theta = x
    ax_b, ay_b, omega = u

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Body -> Global Acceleration
    a_gx = (cos_t * ax_b) + (sin_t * ay_b)
    a_gy = (sin_t * ax_b) + (cos_t * ay_b)

    # Integrate new velocities
    vx_new = vx + a_gx * dt
    vy_new = vy + a_gy * dt

    # Integrate new positions (with constant-acceleration kinematics)
    px_new = px + (vx * dt) + (0.5 * a_gx * dt**2)
    py_new = py + (vy * dt) + (0.5 * a_gy * dt**2)

    # Integrate new orientation
    theta_new = (theta + omega * dt) % (2 * np.pi)

    return np.array([px_new, py_new, vx_new, vy_new, theta_new])

def h_position(x):
    """
    Position-only measurement model
    """
    return x[0:2]

def F_jac_robot(x, u, dt):
    """
    Analytic Jacobian F df/dx for f_robot.
    Returns a 5-by-5 matrix.
    """

    px, py, vx, vy, theta = x
    ax_b, ay_b, omega = u

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Partial derivatives of global accel. wrt theta
    da_gx_dtheta = (-sin_t * ax_b) + (cos_t * ay_b)
    da_gy_dtheta = (cos_t * ax_b) + (sin_t * ay_b)

    F = np.zeros((5, 5))

    # px_new = px + vx*dt + 0.5 * a_gx * dt^2
    F[0, 0] = 1.0
    F[0, 1] = 0.0
    F[0, 2] = dt
    F[0, 3] = 0.0
    F[0, 4] = (0.5 * (dt ** 2)) * da_gx_dtheta

    # py_new = py + (vy * dt) + (0.5 * a_gy * dt**2)
    F[1, 0] = 0.0
    F[1, 1] = 1.0
    F[1, 2] = 0.0
    F[1, 3] = dt
    F[1, 4] = (0.5 * (dt ** 2)) * da_gy_dtheta

    # vx_new = vx + a_gx * dt
    F[2, 0] = 0.0
    F[2, 1] = 0.0
    F[2, 2] = 1.0
    F[2, 3] = 0.0
    F[2, 4] = dt * da_gx_dtheta

    # vy_new = vy + a_gy * dt
    F[3, 0] = 0.0
    F[3, 1] = 0.0
    F[3, 2] = 0.0
    F[3, 3] = 1.0
    F[3, 4] = dt * da_gy_dtheta

    # theta_new = theta + (omega * dt)
    F[4, 0] = 0.0
    F[4, 1] = 0.0
    F[4, 2] = 0.0
    F[4, 3] = 0.0
    F[4, 4] = 1.0

    return F

def H_jac_position(x):
    """
    Jacobian H of model function h
    """
    H = np.zeros((2, 5))
    H[0, 0] = 1.0   # Measurement depends on px
    H[1, 1] = 1.0   # Measuremeant depends on py
    return H

def main():
    """
    Simulation of Extended Kalman Filter for a 2D robot.
    """
    dt = 0.1    # timestep
    steps = 50  # steps

    # Initial state: at origin, no velocity, heading 0
    x0 = np.array([0, 0, 0, 0, 0])
    P0 = np.diag([0.1, 0.1, 0.1, 1.0, 0.1])
    Q = np.diag([1e-4] * 5)
    R = np.diag([0.1, 0.1])

    ekf = ExtendedKalmanFilter(
        x0=x0, P0=P0, Q=Q, R=R, f=f_robot, h=h_position, F_jac=F_jac_robot, H_jac=H_jac_position
    )

    # Control inputs: constant accel forward, no lateral accel, constant yaw rate
    u_forward_accel = 3.5
    u_lat_accel = 0.0
    u_yaw_rate = 0.3
    u = np.array([u_forward_accel, u_lat_accel, u_yaw_rate])

    true_states = []
    est_states = []

    x_true = x0.copy()

    for _ in range(steps):
        # Simulate true robot motion
        x_true = f_robot(x_true, u, dt)
        true_states.append(x_true)

        # Simulate noisy position measurement
        z = h_position(x_true) + np.random.multivariate_normal(mean = [0, 0], cov = R)

        # EKF predict & update
        ekf.predict(u = u, dt = dt)
        ekf.update(z)
        est_states.append(ekf.state())

    true_states = np.array(true_states)
    est_states = np.array(est_states)

    # Plot results
    plt.figure(figsize = (10, 6))
    plt.plot(true_states[:, 0], true_states[:, 1], label = 'True position', linewidth = 2)
    plt.plot(est_states[:, 0], est_states[:, 1], label = 'EKF estimate', linestyle = '--')
    plt.xlabel('X position [m]')
    plt.ylabel('Y position [m]')
    plt.title('EKF Robot Localization')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()