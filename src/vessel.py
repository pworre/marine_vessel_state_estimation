import numpy as np
from dataclasses import dataclass

@dataclass
class VesselParameters:
    """
    Parameters for the 3-DOF vessel model.

    All quantities are expressed in SI units.

    State:
        eta = [x, y, psi]
        nu  = [u, v, r]

    x, y : position in the NED/inertial frame [m]
    psi  : heading [rad]
    u    : surge velocity [m/s]
    v    : sway velocity [m/s]
    r    : yaw rate [rad/s]
    """

    # Inertia / effective mass
    m_u: float = 100_000.0      # surge mass [kg]
    m_v: float = 150_000.0      # sway mass [kg]
    I_r: float = 10_000_000.0   # yaw inertia [kg m^2]

    # Linear damping
    d_u: float = 20_000.0       # surge damping [N s/m]
    d_v: float = 80_000.0       # sway damping [N s/m]
    d_r: float = 10_000_000.0   # yaw damping [N m s/rad]

class Vessel:
    def __init__(self, parameters: VesselParameters | None = None):
        if parameters is None:
            parameters = VesselParameters()

        self.parameters = parameters

        # State
        self.state = np.zeros(6)

    @staticmethod
    def rotation_matrix(psi: float) -> np.ndarray:
        return np.array([
            [np.cos(psi), -np.sin(psi), 0.0],
            [np.sin(psi), np.cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ])

    def kinematics(self, eta: np.ndarray, nu: np.ndarray) -> np.ndarray:
        psi = eta[2]
        R = self.rotation_matrix(psi)

        return R @ nu

    def dynamics(self, nu: np.ndarray, tau: np.ndarray) -> np.ndarray:
        p = self.parameters

        # Inertia matrix
        M = np.diag([
            p.m_u,
            p.m_v,
            p.I_r
        ])

        # Linear damping matrix
        D = np.diag([
            p.d_u,
            p.d_v,
            p.d_r
            ])

        # Calculate acceleration
        nu_dot = np.linalg.solve(
            M,
            tau - D@nu
        )

        return nu_dot

    def state_derivative(self, state: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        State:

            state = [x, y, psi, u, v, r]
        """

        eta = state[:3]
        nu = state[3:]

        eta_dot = self.kinematics(eta, nu)
        nu_dot = self.dynamics(nu, tau)

        return np.concatenate((eta_dot, nu_dot))

    def step(self, tau: np.ndarray, dt: float) -> np.ndarray:
        state_dot = self.state_derivative(self.state, tau)

        self.state += state_dot * dt

        # ! Keep heading within [-pi, pi]
        self.state[2] = self.wrap_angle(self.state[2])

        return self.state.copy()

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float)

        if state.shape != (6,):
            raise ValueError(
                "State must have shape (6,):"
                "[x y psi u v r]"
            )

        self.state = state.copy()

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    @staticmethod
    def wrap_angle(angle:float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

